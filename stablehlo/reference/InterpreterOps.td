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

#ifndef STABLEHLO_REFERENCE_INTERPRETER_OPS
#define STABLEHLO_REFERENCE_INTERPRETER_OPS

include "stablehlo/dialect/Base.td"

//===----------------------------------------------------------------------===//
// Interpreter dialect.
//===----------------------------------------------------------------------===//

def Interpreter_Dialect : Dialect {
  let name = "interpreter";
  let cppNamespace = "::mlir::stablehlo::interpreter";

  let description = [{
    Dialect to implement interpreter-specific functionality, outside of the
    StableHLO spec.
  }];

  let usePropertiesForAttributes = 0;
}

def Interpreter_ArrayOfStrArrayAttr : TypedArrayAttrBase<StrArrayAttr,
    "Array of StrArrayAttr">;

//===----------------------------------------------------------------------===//
// Interpreter op definitions.
//===----------------------------------------------------------------------===//

def Interpreter_RunParallelOp : Op<Interpreter_Dialect, "run_parallel", []> {
  let summary = "RunParallel operation";
  let description = [{
    Runs a two-dimensional grid of `num_replicas` by `num_partitions` StableHLO
    processes based on programs specified by two-dimensional grid of `programs`.

    Given that each program in the grid can have a different number of inputs
    and outputs, we'd want a ragged three-dimensional tensor to model `inputs`
    and `results`, which cannot be easily achieved via conventional means.

    Instead, `inputs` and `results` are represented in a flattened way, obtained
    by reshaping the said three-dimensional tensors into a one-dimensional form
    (with the same semantics as StableHLO's reshape). This is a reversible
    transformation because StableHLO programs have a static number of inputs and
    outputs.

    For example, for a 2x2 StableHLO process grid, `inputs` first has inputs
    from the process `(0, 0)` (i.e. replica_id = 0 and partition_id = 0),
    then `(0, 1)` (i.e. replica_id = 0 and partition_id = 1), etc.

    Example:
    ```mlir
    %results:2 = "interpreter.run_parallel"() {
      programs=[["foo"], ["bar"]]
    } : () -> (tensor<ui32>, tensor<ui32>)
    ```
  }];
  let arguments = (ins
    Variadic<HLO_TensorOrToken>:$inputs,
    Interpreter_ArrayOfStrArrayAttr:$programs
  );
  let results = (outs Variadic<HLO_TensorOrToken>:$results);
  let hasVerifier = 1;
}

#endif  // STABLEHLO_REFERENCE_INTERPRETER_OPS
