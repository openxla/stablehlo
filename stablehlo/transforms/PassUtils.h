/* Copyright 2024 The StableHLO Authors. All Rights Reserved.
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

#ifndef THIRD_PARTY_STABLEHLO_STABLEHLO_TRANSFORMS_PASS_UTILS_H_
#define THIRD_PARTY_STABLEHLO_STABLEHLO_TRANSFORMS_PASS_UTILS_H_

#include "llvm/Support/ErrorHandling.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {
namespace stablehlo {

// Utility functions common across passes.

template <typename T>
Attribute getScalarLike(OpBuilder &b, T constant, Type type) {
  Type element = getElementTypeOrSelf(type);
  if (isa<IntegerType>(element)) return b.getIntegerAttr(element, constant);
  if (isa<FloatType>(element)) return b.getFloatAttr(element, constant);
  if (auto complexTy = dyn_cast<ComplexType>(element)) {
    return complex::NumberAttr::get(complexTy, constant, 0);
  }
  llvm_unreachable("unhandled element type");
}

// Creates a constant with using IntegerAttr, FloatAttr, or ComplexAttr stored
// in `scalar`.
// Returns stablehlo::ConstantOp if value type if static, else returns
// chlo::ConstantLikeOp.
Value getConstantLikeImpl(OpBuilder &b, Location loc, Attribute scalar,
                          Value val);

// Creates a chlo::ConstantLikeOp using a splat `constant` of the same shape
// as `val`.
template <typename T>
Value getConstantLike(OpBuilder &b, Location loc, T constant, Value val) {
  auto shapedTy = cast<ShapedType>(val.getType());
  Attribute scalar = getScalarLike(b, constant, shapedTy);
  return getConstantLikeImpl(b, loc, scalar, val);
}

// Creates a chlo::ConstantLikeOp using a APFloat splat `constant` of the
// same shape as `val`.
// The distinction between double and APFloat causes issues so need this
// explicit template specialization.
Value getConstantLike(OpBuilder &b, Location loc, const APFloat &constant,
                      Value val);

// Check if any of the given types are mlir::quant::QuantizedType.
bool isAnyQuantizedTypes(TypeRange types);

}  // namespace stablehlo
}  // namespace mlir

#endif  // THIRD_PARTY_STABLEHLO_STABLEHLO_TRANSFORMS_PASS_UTILS_H_
