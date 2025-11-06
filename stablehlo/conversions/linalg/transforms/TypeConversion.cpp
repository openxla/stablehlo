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
#include "stablehlo/transforms/conversions/TypeConversion.h"

namespace mlir::stablehlo {

namespace {

Value scalarToTensor(OpBuilder& builder, Type type, ValueRange inputs,
                     Location loc) {
  assert(inputs.size() == 1);
  if (mlir::isa<ShapedType>(inputs.front().getType())) {
    return Value();
  }
  Value result =
      tensor::FromElementsOp::create(
          builder, loc, RankedTensorType::get({}, inputs.front().getType()),
          inputs.front())
          .getResult();
  // Convert to a signed integer if necessary.
  Type elementType = mlir::getElementTypeOrSelf(type);
  if (elementType.isInteger() && !elementType.isSignlessInteger()) {
    result = UnrealizedConversionCastOp::create(builder, loc, type, result)
                 ->getResult(0);
  }
  return result;
}

}  // namespace

LinalgTypeConverter::LinalgTypeConverter() : RemoveSignTypeConverter() {
  addSourceMaterialization(scalarToTensor);
  addTargetMaterialization(scalarToTensor);
}

}  // namespace mlir::stablehlo
