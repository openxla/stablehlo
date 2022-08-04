/* Copyright 2022 The StableHLO Authors.

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

#ifndef STABLEHLO_DIALECT_OPS_H
#define STABLEHLO_DIALECT_OPS_H

#include "llvm/ADT/StringRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"

namespace mlir {
namespace stablehlo {

class StableHLODialect : public Dialect {
 public:
  explicit StableHLODialect(MLIRContext *context);
  static StringRef getDialectNamespace() { return "stablehlo"; }
};

}  // namespace stablehlo
}  // namespace mlir

#define GET_OP_CLASSES
#include "dialect/Ops.h.inc"

#endif  // STABLEHLO_DIALECT_OPS_H
