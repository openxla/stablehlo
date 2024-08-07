/* Copyright 2022 The IREE Authors
   Copyright 2023 OpenXLA Authors. All Rights Reserved.

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

#include "stablehlo/conversions/linalg/transforms/TypeConversion.h"

#include <cassert>
#include <optional>

#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"

namespace mlir::stablehlo {

namespace {

Type convertInteger(IntegerType intType) {
  return IntegerType::get(intType.getContext(),
                          intType.getIntOrFloatBitWidth());
}

Type convertShapedType(ShapedType shapedType) {
  if (auto intType = llvm::dyn_cast<IntegerType>(shapedType.getElementType()))
    return shapedType.clone(convertInteger(intType));
  return shapedType;
}

std::optional<Value> materializeCastFromIllegal(OpBuilder &builder, Type type,
                                                ValueRange inputs,
                                                Location loc) {
  Type fromType = getElementTypeOrSelf(inputs[0].getType());
  Type toType = getElementTypeOrSelf(type);
  if ((!fromType.isSignedInteger() && !fromType.isUnsignedInteger()) ||
      !toType.isSignlessInteger())
    return std::nullopt;
  // Use unrealized conversion casts to do signful->signless conversions.
  return builder.create<UnrealizedConversionCastOp>(loc, type, inputs[0])
      ->getResult(0);
}

std::optional<Value> materializeCastToIllegal(OpBuilder &builder, Type type,
                                              ValueRange inputs, Location loc) {
  Type fromType = getElementTypeOrSelf(inputs[0].getType());
  Type toType = getElementTypeOrSelf(type);
  if (!fromType.isSignlessInteger() ||
      (!toType.isSignedInteger() && !toType.isUnsignedInteger()))
    return std::nullopt;
  // Use unrealized conversion casts to do signless->signful conversions.
  return builder.create<UnrealizedConversionCastOp>(loc, type, inputs[0])
      ->getResult(0);
}

std::optional<Value> scalarToTensor(OpBuilder &builder, Type type,
                                    ValueRange inputs, Location loc) {
  assert(inputs.size() == 1);
  if (mlir::isa<ShapedType>(inputs.front().getType())) {
    return std::nullopt;
  }
  Value result =
      builder
          .create<tensor::FromElementsOp>(
              loc, RankedTensorType::get({}, inputs.front().getType()),
              inputs.front())
          .getResult();
  // Convert to a signed integer if necessary.
  Type elementType = mlir::getElementTypeOrSelf(type);
  if (elementType.isInteger() && !elementType.isSignlessInteger()) {
    result = builder.create<UnrealizedConversionCastOp>(loc, type, result)
                 ->getResult(0);
  }
  return result;
}

}  // namespace

RemoveSignTypeConverter::RemoveSignTypeConverter() {
  addConversion([](Type type) { return type; });

  addConversion(convertInteger);
  addConversion(convertShapedType);

  addArgumentMaterialization(materializeCastFromIllegal);
  addSourceMaterialization(materializeCastToIllegal);
  addTargetMaterialization(materializeCastFromIllegal);
}

LinalgTypeConverter::LinalgTypeConverter() : RemoveSignTypeConverter() {
  addArgumentMaterialization(scalarToTensor);
}

}  // namespace mlir::stablehlo
