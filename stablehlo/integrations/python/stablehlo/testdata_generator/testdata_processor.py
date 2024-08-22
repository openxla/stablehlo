# Copyright 2024 The StableHLO Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Utils to process testdata."""

import re
from typing import Sequence, Set
from absl import logging
from mlir import ir
from mlir import passmanager as pm
from mlir.dialects import check as check_dialect
from mlir.dialects import func as func_dialect
from mlir.dialects import stablehlo as stablehlo_dialect
import numpy as np


def _is_check_op(op: ir.Operation) -> bool:
  """Checks if an MLIR operation is a supported check operation.

  This function identifies whether the given operation is one of the recognized
  check operations,
  including:
    - check.expect_eq
    - check.expect_eq_const
    - check.expect_almost_eq
    - check.expect_almost_eq_const
  Or a stablehlo.custom_call with call_target_name as 'check.eq'

  Args:
    op: The MLIR operation to check.

  Returns:
    True if the operation is a check operation, False otherwise.
  """
  check_ops = [
      check_dialect.ExpectEqOp,
      check_dialect.ExpectEqConstOp,
      check_dialect.ExpectAlmostEqOp,
      check_dialect.ExpectAlmostEqConstOp,
      stablehlo_dialect.CustomCallOp,
  ]

  if any(isinstance(op.opview, check_op) for check_op in check_ops):
    return True

  if isinstance(op.opview, stablehlo_dialect.CustomCallOp):
    return op.opview.call_target_name.value == "check.eq"

  return False


def _get_constant_ops_providing_program_inputs(
    module: ir.Module,
) -> Sequence[ir.Operation]:
  """Identifies and returns the constant operations that provide input values to the program logic in an MLIR module.

  This function analyzes the first function within the provided MLIR module and
  extracts a sequence of StableHLO constant
  operations (`stablehlo.ConstantOp`) whose results are used as inputs to the
  main computational logic of the program.

  Args:
    module: The MLIR module to analyze.

  Returns:
    A list `stablehlo.ConstantOp` operations that are
                             considered to provide input values to the program's
                             main logic.
  """

  return [
      op
      for op in module.body.operations[0].body.blocks[0].operations
      if isinstance(op, stablehlo_dialect.ConstantOp)
      and any(not _is_check_op(use.owner) for use in op.result.uses)
  ]


def _get_ops_under_test(
    module: ir.Module,
) -> Set[ir.Operation]:
  """Identifies and returns the operations under test within an MLIR module.

  This function performs the following steps to extract operations under test:

  1. It calls `_get_constant_ops_providing_program_inputs` to obtain the
  constant
  operations that
     provide input values to the main logic of the program.
  2. For each constant operation, it finds all operations that use its results.
  3. It filters out any operations that are of the following types:
      - `stablehlo_dialect.CustomCallOp`
      - `check.*`
  4. It returns a set of the remaining operations, which are considered the
  operations under test.

  Args:
    module: The MLIR module (in StableHLO dialect) to analyze.

  Returns:
    A set of unique MLIR operations that are identified as being under test.
  """
  constant_ops_for_program_inputs = _get_constant_ops_providing_program_inputs(
      module
  )
  return {
      use.owner
      for const_op in constant_ops_for_program_inputs
      for use in const_op.result.uses
      if not _is_check_op(use.owner)
  }


def _extract_testdata_inputs(
    module: ir.Module,
) -> Sequence[np.ndarray]:
  """Extracts input data (as NumPy arrays) from a StableHLO module.

  It performs the following steps:

  1. Identifies the constant operations within the module that are
     used as input values for the main computation (excluding those used
     for checks or other purposes).
  2. Extracts the numerical values from the dense elements attributes of
     these constant operations.
  3. Converts the extracted values into NumPy arrays.

  Args:
    module: The MLIR module (in StableHLO dialect).

  Returns:
    A sequence of NumPy arrays, where each array corresponds to the input data
    extracted from a constant operation.

  Raises:
      ValueError: If an error occurs during the extraction of input values,
                  such as a mismatch between the number of constant operations
                  and the extracted values.
  """
  constant_ops = _get_constant_ops_providing_program_inputs(module)

  input_values = []
  for constant_op in constant_ops:
    attr = constant_op.opview.value
    if isinstance(attr, ir.DenseElementsAttr):
      input_values.append(np.array(attr))

  if len(input_values) != len(constant_ops):
    raise ValueError("Error in extracting input values")

  return input_values


def _replace_argument_with_constant(
    module: ir.Module,
    inputs: Sequence[np.ndarray],
) -> ir.Module:
  """Replaces arguments of the main function in an MLIR module with constant values.

  The constant values are derived from the provided `inputs` NumPy arrays.
  The function also updates the function signature to reflect the removal of the
  arguments.

  Args:
    module: The MLIR module containing the function whose arguments are to be
      replaced.
    inputs: A list of NumPy arrays, where each array represents the constant
      value for a corresponding function argument.

  Returns:
      ir.Module: The modified MLIR module with the function arguments replaced
      by constants.
  """
  main_func = module.body.operations[0]
  with module.context as ctx, ir.Location.unknown(ctx):
    entry_block = main_func.body.blocks[0]
    with ir.InsertionPoint.at_block_begin(entry_block):
      # Replace function arguments with constants for the input values
      for input in inputs:
        const_op = stablehlo_dialect.ConstantOp(ir.DenseElementsAttr.get(input))
        entry_block.arguments[0].replace_all_uses_with(const_op.result)
        entry_block.erase_argument(0)

    # Update the type  of the entry function.
    main_ftype = ir.FunctionType.get([], main_func.type.results)
    main_func.function_type = ir.TypeAttr.get(main_ftype)

  return module


def is_testdata_format(module: ir.Module) -> bool:
  """Checks if an MLIR module has one function with no arguments and contains a check operation.

  Args:
    module: The MLIR module to be verified.

  Raises:
      AssertionError: If the module fails on any of the above criterias.
  """
  functions = [
      op for op in module.body.operations if isinstance(op, func_dialect.FuncOp)
  ]
  if len(functions) != 1:
    func_names = [func.name for func in functions]
    raise AssertionError(
        "Testdata format expected to have module with one function, but got"
        f" {func_names}."
    )

  main_func = functions[0]
  if len(main_func.body.blocks) != 1:
    raise AssertionError(
        "Testdata format expected to have the main function with a single"
        f" block, but got {len(main_func.body.blocks)} blocks."
    )

  check_op_available = any(
      _is_check_op(op.operation) for op in main_func.body.blocks[0].operations
  )
  return check_op_available and not main_func.type.inputs


def to_testdata_format(
    module: ir.Module,
    inputs: Sequence[np.ndarray],
    golden_results: Sequence[np.ndarray],
) -> ir.Module:
  """Transforms an MLIR module to testdata format.

  Transforms `module` with the following modfications:
    - Removes Function Arguments: It eliminates the function arguments from the
      function definition.
    - Introduces Constant Operations: It replaces those arguments with
      stablehlo.constant operations on input values `inputs`.
    - Maintains operation under test: The operation under test of the function
      remains unchanged. It now takes the newly created constant values as
      operands.
    - Adds Check Operation: A new stablehlo.custom_call operation with a
      call_target_name of "check.eq" is added to the function body.
      This operation compares the result of the operation under test with the
      constant value `golden_results`. The result of this comparison is a tensor
      of booleans indicating where the results match the expected values.
    - Adjusts Function Return: The function's return type is updated to
      reflect the fact that it now returns the result of stablehlo.custom_call
      operation.

    Args:
      module: The MLIR module containing the stablehlo operations.
      inputs: A list of NumPy arrays representing the input data.
      golden_results: A list of NumPy arrays representing the expected outputs
        of executing `module` on `inputs`.

    Returns:
    A new MLIR module with with transformation mentioned above.

  Example Input:

  - `module`:
    module {
      func.func public @main(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) ->
          tensor<2xf32> {
        %0 = stablehlo.add %arg0, %arg1 : tensor<2xf32>
        return %0 : tensor<2xf32>
      }
    }
  - inputs`: [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
  - golden_results: [np.array([4.0, 6.0])]

  Example Output:
    module {
      func.func @add_op_test_f32() -> tensor<2xf32> {
        %0 = stablehlo.constant dense<[1.0, 2.0]> : tensor<2xf32>
        %1 = stablehlo.constant dense<[3.0, 4.0]> : tensor<2xf32>
        %2 = stablehlo.add %0, %1 : tensor<2xf32>
        %3 = stablehlo.custom_call("check.eq", %2, dense<[4.0, 6.0]> :
          tensor<2xi1>
        func.return %2 : tensor<2xf32>
      }
    }
  """

  with module.context as ctx, ir.Location.unknown(ctx):
    entry_block = module.body.operations[0].body.blocks[0]
    with ir.InsertionPoint.at_block_begin(entry_block):
      # Replace function arguments with constants for the input values
      for input in inputs:
        const_op = stablehlo_dialect.ConstantOp(ir.DenseElementsAttr.get(input))
        entry_block.arguments[0].replace_all_uses_with(const_op.result)
        entry_block.erase_argument(0)

      # Create constant ops for golden results
      golden_result_constants = [
          stablehlo_dialect.ConstantOp(ir.DenseElementsAttr.get(golden_result))
          for golden_result in golden_results
      ]

    # Find the original return operation
    return_op = entry_block.operations[len(entry_block.operations) - 1]
    if not isinstance(return_op, func_dialect.ReturnOp):
      raise AssertionError(
          "Expects the last operation in function block to be a return op, but"
          f" got: {return_op}"
      )
    return_operands = return_op.operands

    # Insert check operations at the end of the block, just before the return
    with ir.InsertionPoint.at_block_terminator(entry_block):
      for idx, operand in enumerate(return_operands):
        custom_call = stablehlo_dialect.CustomCallOp(
            [ir.RankedTensorType.get([], ir.IntegerType.get_signless(1))],
            [golden_result_constants[idx].result, operand],
            call_target_name="check.eq",
        )

    # Update the function's type to reflect the new return values
    main_ftype = ir.FunctionType.get(
        [], [return_operand.type for return_operand in return_operands]
    )
    module.body.operations[0].function_type = ir.TypeAttr.get(main_ftype)

    # Verify the module is valid and in testdata format
    assert module.operation.verify()
    assert is_testdata_format(module)

    return module


def from_testdata_format(module: ir.Module) -> ir.Module:
  """Transforms `module` in testdata format.

  Transforms `module` with the following modfications:
    - The constants in the `module` will be replaced with arguments of the main
      function in `module`.
    - The transformed module returns the results of the the operations under
    test.

  Args:
    module: The original MLIR module containing the `op_under_test`.
    op_under_test: An operation in `module`.

  Returns:
    A new MLIR module with transformation mentioned above.

  Example Input:

  - `module`:
    module {
      func.func @add_op_test_f32() {
        %0 = stablehlo.constant dense<[1.0, 2.0]> : tensor<2xf32>
        %1 = stablehlo.constant dense<[3.0, 4.0]> : tensor<2xf32>
        %2 = stablehlo.add %0, %1 : tensor<2xf32>
        check.expect_eq_const %2, dense<[4.0, 6.0]> : tensor<2xf32>
        func.return
      }
    }
  - `op_under_test`: %2 = stablehlo.add %0, %1 : tensor<2xf32>

  Example Output:

    module {
      func.func public @main(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) ->
          tensor<2xf32> {
        %0 = stablehlo.add %arg0, %arg1 : tensor<2xf32>
        return %0 : tensor<2xf32>
      }
    }
  """

  if not is_testdata_format(module):
    return module

  with module.context as ctx, ir.Location.unknown(ctx) as loc:
    main_func = module.body.operations[0]
    module_ops = main_func.body.blocks[0].operations

    # Extract constant ops used as inputs for the program (not for check ops)
    constant_ops_for_program_inputs = [
        op
        for op in module_ops
        if isinstance(op, stablehlo_dialect.ConstantOp)
        and any(not _is_check_op(use.owner) for use in op.result.uses)
    ]

    # Extract check operations
    check_ops = [op for op in module_ops if _is_check_op(op)]

    # Extract constant ops feeding into check ops
    constant_ops_feeding_to_check_ops = [
        op
        for op in module_ops
        if isinstance(op, stablehlo_dialect.ConstantOp)
        and all(_is_check_op(use.owner) for use in op.result.uses)
    ]

    # Extract non-constant ops feeding into check ops
    non_constant_ops_feeding_to_check_ops = [
        operand.owner
        for check_op in check_ops
        for operand in check_op.operands
        if not isinstance(operand.owner.opview, stablehlo_dialect.ConstantOp)
    ]

    # Remove unused ops that feed into check ops
    ops_to_remove = constant_ops_feeding_to_check_ops + check_ops

    # Update the main function's type based on inputs and original output types
    input_types = [
        value.result.type for value in constant_ops_for_program_inputs
    ]
    result_types = [
        check_op.operands[0].owner.result.type for check_op in check_ops
    ]

    # Update the function signature with the derived input and result types
    main_ftype = ir.FunctionType.get(input_types, result_types)
    main_func.function_type = ir.TypeAttr.get(main_ftype)

    # Replace constants with function arguments
    entry_block = main_func.body.blocks[0]
    for idx, constant_op in enumerate(constant_ops_for_program_inputs):
      arg = entry_block.add_argument(constant_op.result.type, loc)
      constant_op.result.replace_all_uses_with(arg)
      constant_op.erase()

    # Validate that the last operation is a return operation
    return_op = entry_block.operations[len(entry_block.operations) - 1]
    if not isinstance(return_op, func_dialect.ReturnOp):
      raise AssertionError(
          "Expects the last operation in function block to be a return op, but"
          f" got: {return_op}"
      )

    # Remove unused ops in reverse order to avoid invalidating indices
    ops_to_remove.append(return_op)
    ops_to_remove.reverse()
    [op.erase() for op in ops_to_remove]

    # Update the return statement to return the results of the non-constant operations
    with ir.InsertionPoint(entry_block):
      func_dialect.ReturnOp(non_constant_ops_feeding_to_check_ops)

    # Verify that the module is valid after the transformations
    assert module.operation.verify()

    return module


def preprocess_input_module(module: ir.Module):
  """Preprocesses a StableHLO module in testdata format.

  This function performs the following key steps:

  1. Extracts the operation under test and its input values as NumPy arrays.
  2. Transforms the module to isolate the operation under test and make its
  operands into function arguments.

  Args:
    module: MLIR module in text format.

  Returns:
    A tuple containing:
     - The preprocessed MLIR module with the isolated operation under test.
     - A list of NumPy arrays representing the input values for the operation.

  Raises:
      AssertionError: If the module structure fails to comply with testdata
      format.
  """
  inputs = _extract_testdata_inputs(module)
  module = from_testdata_format(module)
  return module, inputs
