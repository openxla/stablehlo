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
}

//===----------------------------------------------------------------------===//
// CHECK op definitions.
//===----------------------------------------------------------------------===//

def CHECK_AlmostEqOp :
    Op<CHECK_Dialect,
       "almost_eq", [AllTypesMatch<["lhs", "value"]>]> {
  let summary = [{Checks the tensor operand is almost equal to some constant}];
  let description =  [{
    Verifies that the tensor operand with floating-point or complex element
    type is almost equal to the constant attribute within an
    implementation-defined tolerance.

    ```mlir
    check.almost_eq %arg0, dense<[0.999999, 2.0]> : tensor<2xf32>
    ```
  }];

  let arguments = (ins
    TensorOf<[AnyFloat, AnyComplex]>:$lhs,
    ElementsAttr:$value
  );

  let assemblyFormat = "$lhs `,` $value attr-dict";
}

def CHECK_EqOp :
    Op<CHECK_Dialect, "eq", [AllTypesMatch<["lhs", "value"]>]> {
  let summary = [{Checks the tensor operand is equal to some constant}];
  let description =  [{
    Verifies that the tensor operand is exactly equal to a constant attribute.

    ```mlir
    check.eq %arg0, dense<[1, 2]> : tensor<2xi32>
    ```
  }];

  let arguments = (ins
    HLO_Tensor:$lhs,
    ElementsAttr:$value
  );

  let assemblyFormat = "$lhs `,` $value attr-dict";
}

#endif  // STABLEHLO_DIALECT_CHECK_OPS
