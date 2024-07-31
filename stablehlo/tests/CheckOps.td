/* Copyright 2023 The StableHLO Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef STABLEHLO_DIALECT_CHECK_OPS
#define STABLEHLO_DIALECT_CHECK_OPS

include "stablehlo/dialect/Base.td"

//===----------------------------------------------------------------------===//
// CHECK dialect.
//===----------------------------------------------------------------------===//

def CHECK_Dialect : Dialect {
  let name = "check";
  let cppNamespace = "::mlir::stablehlo::check";

  let description = [{
    Opset to implement checks for testing the StableHLO interpreter.
  }];

  let useDefaultAttributePrinterParser = 0;
  let useDefaultTypePrinterParser = 0;
  let usePropertiesForAttributes = 1;
}

//===----------------------------------------------------------------------===//
// CHECK op definitions.
//===----------------------------------------------------------------------===//

def CHECK_ExpectAlmostEqConstOp :
    Op<CHECK_Dialect,
       "expect_almost_eq_const", [AllTypesMatch<["lhs", "value"]>]> {
  let summary = [{Checks the tensor operand is almost equal to some constant}];
  let description =  [{
    Verifies that the tensor operand with floating-point or complex element
    type is almost equal to the constant attribute within an
    implementation-defined tolerance.

    ```mlir
    check.expect_almost_eq_const %arg0, dense<[0.999999, 2.0]> : tensor<2xf32>, tolerance = 0.0001
    ```
  }];

  let arguments = (ins
    TensorOf<[AnyFloat, AnyComplex]>:$lhs,
    ElementsAttr:$value,
    DefaultValuedOptionalAttr<F64Attr, "1e-4">:$tolerance
  );

  let assemblyFormat = "$lhs `,` $value (`,` `tolerance` `=` $tolerance^)? attr-dict";
}

def CHECK_ExpectAlmostEqOp :
    Op<CHECK_Dialect, "expect_almost_eq", [SameTypeOperands]> {
  let summary = [{Checks that the tensor operands are almost equal}];
  let description = [{
    Verifies that the tensor operands with floating-point or complex element
    types are almost equal within an implementation-defined tolerance.

    ```mlir
    check.expect_almost_eq %arg0, %arg1, tolerance = 0.001 : f64 : tensor<2xf32>
    ```
  }];

  let arguments = (ins
      TensorOf<[AnyFloat, AnyComplex]>:$lhs,
      TensorOf<[AnyFloat, AnyComplex]>:$rhs,
      DefaultValuedOptionalAttr<F64Attr, "1e-4">:$tolerance
  );

  let assemblyFormat = "$lhs `,` $rhs (`,` `tolerance` `=` $tolerance^)? attr-dict `:` type($lhs)";
}

def CHECK_ExpectEqConstOp :
    Op<CHECK_Dialect, "expect_eq_const", [AllTypesMatch<["lhs", "value"]>]> {
  let summary = [{Checks the tensor operand is equal to some constant}];
  let description =  [{
    Verifies that the tensor operand is exactly equal to a constant attribute.

    ```mlir
    check.expect_eq_const %arg0, dense<[1, 2]> : tensor<2xi32>
    ```
  }];

  let arguments = (ins
    HLO_Tensor:$lhs,
    ElementsAttr:$value
  );

  let assemblyFormat = "$lhs `,` $value attr-dict";
}

def CHECK_ExpectEqOp : Op<CHECK_Dialect, "expect_eq", [SameTypeOperands]> {
  let summary = [{Checks that the tensor operands are equal}];
  let description = [{
    Verifies that the operands are exactly equal.

    ```mlir
    check.expect_eq %arg0, %arg1 : tensor<2xi32>
    ```
  }];

  let arguments = (ins
      HLO_Tensor:$lhs,
      HLO_Tensor:$rhs
  );

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs)";
}

def CHECK_ExpectCloseOp :
    Op<CHECK_Dialect, "expect_close", [
      AllTypesMatch<["actual", "expected"]>,
      AllShapesMatch<["actual", "expected"]>]> {
  let summary = [{Checks that the tensor operand ULP-distance to expected tensor operand is within specified limits}];
  let description = [{

    Verifies that the floating-point tensor operands, actual and
    expected, are close within user-defined limits:

      min_ulp_difference <= ulp_difference(actual, reference) <= max_ulp_difference

    where ulp_difference returns the number of all possible floating
    point values that are greater than or equal to min(actual,
    expected) and smaller than max(actual, expected) provided that
    both operands are finite.

    When either of operands is non-finite, expect_close returns true
    when actual and expected are bitwise equal, or when both values
    are NaNs, otherwise false. Note that quiet NaN and signaling NaN
    values with different payloads are all considered equivalent.

    ```mlir
    check.expect_close %arg0, %arg1, max_ulp_difference = 3, min_ulp_difference = 0 : tensor<2xf32>, tensor<2xf32>
    ```
  }];

  let arguments = (ins
      TensorOf<[AnyFloat, AnyComplex]>:$actual,
      TensorOf<[AnyFloat, AnyComplex]>:$expected,
      DefaultValuedAttr<UI64Attr, "1">:$max_ulp_difference,
      DefaultValuedOptionalAttr<UI64Attr, "0">:$min_ulp_difference
  );

  let assemblyFormat = [{
      $actual `,` $expected
      `,` `max_ulp_difference` `=` $max_ulp_difference
      (`,` `min_ulp_difference` `=` $min_ulp_difference^)?
      `:` attr-dict type($actual) `,` type($expected)
    }];
}

def CHECK_ExpectSerializedEqOp : Op<CHECK_Dialect, "expect_serialized_eq", []> {
  let summary = [{Checks value of serialized tensor value.}];
  let description = [{
    Verifies that the value and type of the serialized tensor `probe_id` match
    the optionally specified input tensor at iteration `iteration`, using
    previously serialized filepaths in `index.csv`.

    ```mlir
    check.expect_serialized_eq %arg0,
      probe_id = "probe0",
      iter = 0 : tensor<2xi32>
    ```
  }];

  let arguments = (ins
      HLO_Tensor:$expected,
      StrAttr:$probe_id,
      DefaultValuedOptionalAttr<UI32Attr, "0">:$iteration
  );

  let assemblyFormat = [{
    $expected `,` `probe_id` `=` $probe_id (`,` `iter` `=` $iteration^)?
    attr-dict `:` type($expected)
  }];
}

#endif  // STABLEHLO_DIALECT_CHECK_OPS
