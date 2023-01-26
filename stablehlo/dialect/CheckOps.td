/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.
   Copyright 2023 The StableHLO Authors.

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

include "mlir/IR/OpBase.td"
include "stablehlo/dialect/Base.td"

def CHECK_Dialect : Dialect {
  let name = "check";
  let cppNamespace = "::mlir::stablehlo";

  let summary = [{
    A dialect implementing test assertions for StableHLO interpreter.
  }];
}

def CHECK_ExpectEqOp : Op<CHECK_Dialect, "expect_eq"> {
  let summary = [{Checks that the tensor or buffer view operands are equal}];
  let description = [{
    Verifies that the operands are exactly equal.

    Issues a non-fatal failure if the verification fails.

    ```mlir
    check.expect_eq(%arg0, %arg1) : tensor<5xi32>
    ```
  }];

  let arguments = (ins
      HLO_Tensor:$lhs,
      ElementsAttr:$rhs
  );

  let assemblyFormat = "`(` $lhs `,` $rhs `)` attr-dict `:` type($lhs)";
}

#endif  // STABLEHLO_DIALECT_CHECK_OPS
