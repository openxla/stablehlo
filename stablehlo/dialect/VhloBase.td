/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.
   Copyright 2022 The StableHLO Authors.

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

#ifndef STABLEHLO_DIALECT_VHLO_BASE
#define STABLEHLO_DIALECT_VHLO_BASE

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// VHLO Type Definitions.
//===----------------------------------------------------------------------===//

// VHLO is intended to represent the layout only, as such uses AnyType everywhere.
def VHLO_AnyType : AnyTypeOf<[AnyType]>;
def VHLO_AnyAttr : AnyAttrOf<[AnyAttr]>;
def VHLO_AnyRegion : Region<CPred<"true">, "any region">;

// Data types used in Attributes / Types
def VHLO_Dims : ArrayRefParameter<"int64_t", "Dimension"> {
  let parser = "mlir::hlo::parseDimSizes($_parser)";
  let printer = "mlir::hlo::printDimSizes($_printer, $_self)";
}

//===----------------------------------------------------------------------===//
// VHLO traits
//===----------------------------------------------------------------------===//

// VHLO intentionally does not include traits since it is only used to represent
// the in-memory structure of the IR at a given version.
//
// This section is included for file parity between VhloBase.td and Base.td

#endif // STABLEHLO_DIALECT_VHLO_BASE
