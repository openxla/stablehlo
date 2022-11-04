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

#ifndef STABLEHLO_COMPATIBILITY_VERSIONED_STABLEHLO_OPS_H
#define STABLEHLO_COMPATIBILITY_VERSIONED_STABLEHLO_OPS_H

#include <algorithm>

#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/TensorEncoding.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "stablehlo/dialect/Base.h"

// Include order matters.
//#include "stablehlo/dialect/StablehloEnums.h.inc"
//#define GET_ATTRDEF_CLASSES
//#include "stablehlo/dialect/StablehloAttrs.h.inc"

namespace mlir {
namespace versioned_stablehlo {

class StablehloV1Dialect : public Dialect {
 public:
  explicit StablehloV1Dialect(MLIRContext *context);
  static StringRef getDialectNamespace() { return "versioned_stablehlo"; }
};

}  // namespace versioned_stablehlo
}  // end namespace mlir

#define GET_OP_CLASSES
#include "stablehlo/compatibility/VersionedStablehloOps.h.inc"

#endif  // STABLEHLO_COMPATIBILITY_VERSIONED_STABLEHLO_OPS_H
