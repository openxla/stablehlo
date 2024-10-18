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

#include "stablehlo/transforms/PassUtils.h"

#include "llvm/Support/ErrorHandling.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/ChloOps.h"

namespace mlir {
namespace stablehlo {

namespace {
// Need some extra handling to generate a DenseElementsAttr from a complex
// scalar, so add a helper function.
DenseElementsAttr getSplatFromScalar(OpBuilder &b, Attribute scalar,
                                     ShapedType type) {
  if (auto complexScalar = dyn_cast<complex::NumberAttr>(scalar)) {
    return DenseElementsAttr::get(type, complexScalar.getValue());
  }
  return DenseElementsAttr::get(type, scalar);
}
}  // namespace

// Returns `stablehlo::ConstantOp` if value type if static,
// else returns `chlo::ConstantLikeOp`.
Value getConstantLikeImpl(OpBuilder &b, Location loc, Attribute scalar,
                          Value val) {
  if (!llvm::isa<IntegerAttr, FloatAttr, complex::NumberAttr>(scalar))
    llvm::report_fatal_error("unhandled constant like element type");

  auto shapedTy = cast<ShapedType>(val.getType());
  if (shapedTy.hasStaticShape()) {
    Attribute splat = getSplatFromScalar(b, scalar, shapedTy);
    return b.create<mlir::stablehlo::ConstantOp>(loc, splat);
  }

  return b.create<mlir::chlo::ConstantLikeOp>(loc, cast<TypedAttr>(scalar),
                                              val);
}

Value getConstantLike(OpBuilder &b, Location loc, const APFloat &constant,
                      Value val) {
  Type ty = getElementTypeOrSelf(val.getType());
  return getConstantLikeImpl(b, loc, b.getFloatAttr(ty, constant), val);
}

bool isAnyQuantizedTypes(TypeRange types) {
  return llvm::any_of(types, [](Type type) {
    return isa<quant::QuantizedType>(getElementTypeOrSelf(type));
  });
}

}  // namespace stablehlo
}  // namespace mlir
