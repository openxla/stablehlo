/* Copyright 2025 The OpenXLA Authors.

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

#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

#include <cstdint>
#include <optional>

#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/dialect/TypeInference.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/transforms/StablehloBroadcastLowering.h"

namespace mlir {
namespace stablehlo {

///////////////
// Dialect Helpers
///////////////

MlirOp AttachFrontendAttribute(MlirBuilder& builder, MlirOp op, StringRef name,
                               Attribute value) {
  constexpr char kFrontendAttrName[] = "mhlo.frontend_attributes";
  Operation* mlirOp = unwrap(op).getDefiningOp();
  SmallVector<NamedAttribute> attrs;
  DictionaryAttr frontendAttr =
      mlirOp->getAttrOfType<DictionaryAttr>(kFrontendAttrName);
  if (frontendAttr) {
    for (NamedAttribute attr : frontendAttr.getValue()) {
      // Populate all non-conflicting names.
      if (attr.getName() != name) {
        attrs.push_back(attr);
      }
    }
  }
  attrs.emplace_back(name, value);
  mlirOp->setAttr(kFrontendAttrName,
                  DictionaryAttr::get(&builder.getContext(), attrs));
  return op;
}

/////////////////
// MANUAL APIs
/////////////////

MlirOp ConvertElementType(MlirOp input, Type resultElementType) {
  MlirOp operand = input;
  auto inputType = mlir::cast<RankedTensorType>(input.getType());
  auto resultType = inputType.clone(resultElementType);
  if (inputType == resultType) return input;  // skip no-op convert
  if (isa<ComplexType>(inputType.getElementType()) &&
      !isa<ComplexType>(resultElementType)) {
    operand = stablehlo::Real(operand);
  }
  return stablehlo::Convert(resultType, operand);
}

MlirOp ConvertElementType(MlirOp input, ElementType resultElementTypeKind) {
  auto resultElementType =
      getElementType(input.getContext(), resultElementTypeKind);
  return ConvertElementType(input, resultElementType);
}

// These are not finalized APIs, just an example of hiding the ugly for
// important leaf ops.
MlirOp Constant(MlirBuilder& builder, int64_t value) {
  return builder.create<stablehlo::ConstantOp>(DenseIntElementsAttr::get(
      RankedTensorType::get({}, builder.getOpBuilder().getI64Type()), value));
}
MlirOp Constant(MlirBuilder& builder, std::vector<int64_t> value) {
  auto numel = static_cast<int64_t>(value.size());
  return builder.create<stablehlo::ConstantOp>(DenseIntElementsAttr::get(
      RankedTensorType::get({numel}, builder.getOpBuilder().getI64Type()),
      value));
}

MlirOp IotaLike(MlirOp input, int64_t dim, Type elementType) {
  auto inputType = mlir::cast<RankedTensorType>(input.getType());
  if (inputType.hasStaticShape()) {
    return stablehlo::Iota(input.getBuilder(), inputType.clone(elementType),
                           dim);
  }

  // Use input's static shape and slice to the dynamic shape.
  auto dims = mlir::stablehlo::getDimensions(input.getValue());
  if (mlir::failed(dims))
    llvm::report_fatal_error(
        "failed to create dynamically shaped iota op, with MLIR error: ");

  mlir::SmallVector<int64_t> iotaShape = llvm::map_to_vector(
      *dims,
      [&](mlir::stablehlo::DimensionInfo dim_size) { return dim_size.size; });
  auto iotaType =
      mlir::makeTensorType(input.getContext(), iotaShape, elementType);
  mlir::MlirOp iota = mlir::stablehlo::Iota(input.getBuilder(), iotaType, dim);

  // Slice bounded dimensions to the dynamic shape.
  for (const mlir::stablehlo::DimensionInfo& dim : *dims) {
    if (!dim.boundOp.has_value()) continue;

    auto runtime_dim_size =
        mlir::stablehlo::GetDimensionSize(input, dim.boundOpDim);
    iota = mlir::stablehlo::SetDimensionSize(iota, runtime_dim_size,
                                             dim.boundOpDim);
  }
  return iota;
}

MlirOp IotaLike(MlirOp input, int64_t dim, ElementType elementType) {
  auto resultElementType = getElementType(input.getContext(), elementType);
  return IotaLike(input, dim, resultElementType);
}

namespace {

// Use preferred element type, if not use LHS element type.
Type getDotResultType(RankedTensorType lhsType, ShapedTypeComponents type,
                      std::optional<ElementType> preferredResultType) {
  Type elementType =
      preferredResultType.has_value()
          ? getElementType(*lhsType.getContext(), preferredResultType.value())
          : lhsType.getElementType();
  return RankedTensorType::get(type.getDims(), elementType,
                               type.getAttribute());
}

}  // namespace

MlirOp Dot(MlirOp lhs, MlirOp rhs, ArrayAttr precisionConfig,
           std::optional<ElementType> preferredElementType) {
  SmallVector<ShapedTypeComponents> inferredShape;
  auto lhsType = mlir::dyn_cast<RankedTensorType>(lhs.getType());
  auto rhsType = mlir::dyn_cast<RankedTensorType>(rhs.getType());
  if (!lhsType || !rhsType)
    llvm::report_fatal_error(
        "Failed to infer dot op type from lhs and rhs types.");

  if (failed(hlo::inferDotOp(lhs.getValue().getLoc(), lhsType, rhsType,
                             precisionConfig, inferredShape)) ||
      inferredShape.size() != 1)
    llvm::report_fatal_error(
        "Failed to infer dot op type from lhs and rhs types.");

  auto resultType =
      getDotResultType(lhsType, inferredShape[0], preferredElementType);
  return stablehlo::Dot(resultType, lhs, rhs, precisionConfig);
}

MlirOp DotGeneral(MlirOp lhs, MlirOp rhs, DotDimensionNumbersAttr dotDimsAttr,
                  ArrayAttr precisionConfig,
                  std::optional<ElementType> preferredElementType) {
  SmallVector<ShapedTypeComponents> inferredShape;
  auto lhsType = mlir::dyn_cast<RankedTensorType>(lhs.getType());
  auto rhsType = mlir::dyn_cast<RankedTensorType>(rhs.getType());
  if (!lhsType || !rhsType)
    llvm::report_fatal_error(
        "Failed to infer dot op type from lhs and rhs types.");

  if (failed(hlo::inferDotGeneralOp(lhs.getValue().getLoc(), lhsType, rhsType,
                                    dotDimsAttr.getLhsBatchingDimensions(),
                                    dotDimsAttr.getRhsBatchingDimensions(),
                                    dotDimsAttr.getLhsContractingDimensions(),
                                    dotDimsAttr.getRhsContractingDimensions(),
                                    precisionConfig, inferredShape)) ||
      inferredShape.size() != 1)
    llvm::report_fatal_error(
        "Failed to infer dot op type from lhs and rhs types.");

  auto resultType =
      getDotResultType(lhsType, inferredShape[0], preferredElementType);
  return stablehlo::DotGeneral(resultType, lhs, rhs, dotDimsAttr,
                               precisionConfig);
}

MlirOp Reshape(MlirOp input, ArrayRef<int64_t> newShape) {
  auto type = mlir::dyn_cast<RankedTensorType>(input.getType());
  if (!type)
    llvm::report_fatal_error("expected ranked tensor input to reshape");
  auto newType = type.clone(newShape);
  return stablehlo::Reshape(newType, input);
}

SmallVector<MlirOp> Arguments(RegionBuilder& rb, WhileOp op) {
  // Already init arguments, just return.
  if (!rb.getRegion().getArguments().empty()) {
    return wrap(rb, rb.getRegion().getArguments());
  }
  SmallVector<MlirOp> operands;
  operands.reserve(op.getOperands().size());
  for (auto operand : op.getOperands()) {
    operands.push_back(mlir::Argument(rb, operand.getType()));
  }
  return operands;
}

/////////////////
// GENERATED APIs
/////////////////

#include "stablehlo/integrations/cpp/builder/StablehloBuilder.cpp.inc"

}  // namespace stablehlo
}  // namespace mlir
