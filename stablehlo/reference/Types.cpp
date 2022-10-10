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

#include "stablehlo/reference/Types.h"

#include "mlir/IR/BuiltinTypes.h"

namespace mlir {
namespace stablehlo {

bool isSupportedUnsignedIntegerType(Type type) {
  return type.isUnsignedInteger(4) || type.isUnsignedInteger(8) ||
         type.isUnsignedInteger(16) || type.isUnsignedInteger(32) ||
         type.isUnsignedInteger(64);
}

bool isSupportedSignedIntegerType(Type type) {
  // TODO(#22): StableHLO, as bootstrapped from MHLO, inherits signless
  // integers which was added in MHLO for legacy reasons. Going forward,
  // StableHLO will adopt signfull integer semantics with signed and unsigned
  // integer variants.
  return type.isSignlessInteger(4) || type.isSignlessInteger(8) ||
         type.isSignlessInteger(16) || type.isSignlessInteger(32) ||
         type.isSignlessInteger(64);
}

bool isSupportedBooleanType(Type type) { return type.isSignlessInteger(1); }

bool isSupportedIntegerType(Type type) {
  return isSupportedUnsignedIntegerType(type) ||
         isSupportedSignedIntegerType(type);
}

bool isSupportedFloatType(Type type) {
  return type.isF16() || type.isBF16() || type.isF32() || type.isF64();
}

bool isSupportedComplexType(Type type) {
  auto complexTy = type.dyn_cast<ComplexType>();
  if (!complexTy) return false;

  auto complexElemTy = complexTy.getElementType();
  return complexElemTy.isF32() || complexElemTy.isF64();
}

}  // namespace stablehlo
}  // namespace mlir
