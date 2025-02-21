/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

#include "stablehlo/dialect/ChloOps.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <optional>
#include <string>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Region.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/TypeID.h"
#include "mlir/Transforms/InliningUtils.h"
#include "stablehlo/dialect/AssemblyFormat.h"
#include "stablehlo/dialect/Base.h"
#include "stablehlo/dialect/BroadcastUtils.h"
#include "stablehlo/dialect/ChloBytecode.h"
#include "stablehlo/dialect/TypeInference.h"

// Include order matters
#include "stablehlo/dialect/ChloEnums.cpp.inc"
#define GET_ATTRDEF_CLASSES
#include "stablehlo/dialect/ChloAttrs.cpp.inc"

namespace mlir {
namespace chlo {

//===----------------------------------------------------------------------===//
// CompatibleOperandsAndResultType
//===----------------------------------------------------------------------===//

// TODO(b/231358795): Review the use of InferTypeOpInterface for ops that
// support quantization or sparsity.
#define INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(Op)                \
  LogicalResult Op::inferReturnTypeComponents(                        \
      MLIRContext* context, std::optional<Location> location,         \
      ValueShapeRange operands, DictionaryAttr attributes,            \
      OpaqueProperties properties, RegionRange regions,               \
      SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {  \
    return inferReturnTypeComponentsFromOperands(                     \
        context, location, operands, attributes, properties, regions, \
        inferredReturnShapes);                                        \
  }

INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(AsinAcosKernelOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(AcosOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(AcoshOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(AsinOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(AsinhOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(AtanOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(AtanhOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(BesselI1eOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(ConjOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(CoshOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(DigammaOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(ErfOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(ErfcOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(ErfInvOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(LgammaOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(NextAfterOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(PolygammaOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(SinhOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(SquareOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(TanOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(ZetaOp)

//===----------------------------------------------------------------------===//
// BinaryOps
//===----------------------------------------------------------------------===//

namespace {
// Gets the resulting type from a broadcast between two types.
ShapedTypeComponents getBroadcastType(
    Type x, Type y, Type elementType,
    std::optional<ArrayRef<int64_t>> broadcastDimensionsAttr) {
  auto xRanked = dyn_cast<RankedTensorType>(x);
  auto yRanked = dyn_cast<RankedTensorType>(y);
  if (!xRanked || !yRanked) return {elementType};

  auto shapeX = xRanked.getShape();
  auto shapeY = yRanked.getShape();

  // If no broadcast dimensions, assume "numpy" broadcasting.
  if (shapeX.size() == shapeY.size() || !broadcastDimensionsAttr.has_value()) {
    llvm::SmallVector<int64_t, 4> outShape;
    if (!mlir::OpTrait::util::getBroadcastedShape(shapeX, shapeY, outShape)) {
      // Signal illegal broadcast_dimensions as unranked.
      return {elementType};
    }
    return {outShape, elementType};
  }

  auto shapeLarge = shapeX.size() > shapeY.size() ? shapeX : shapeY;
  auto shapeSmall = shapeX.size() <= shapeY.size() ? shapeX : shapeY;

  auto broadcastDimensions = broadcastDimensionsAttr.value();
  if (broadcastDimensions.size() != shapeSmall.size()) {
    // Signal illegal broadcast_dimensions as unranked.
    return {elementType};
  }
  llvm::SmallVector<int64_t, 4> shapeLargeFiltered;
  shapeLargeFiltered.reserve(shapeSmall.size());
  for (const auto& dim : broadcastDimensions) {
    if (dim >= static_cast<int64_t>(shapeLarge.size())) return {elementType};
    shapeLargeFiltered.push_back(shapeLarge[dim]);
  }
  llvm::SmallVector<int64_t, 4> outShapeFiltered;
  if (!mlir::OpTrait::util::getBroadcastedShape(shapeSmall, shapeLargeFiltered,
                                                outShapeFiltered))
    // Signal illegal broadcast_dimensions as unranked.
    return {elementType};

  // Update according to the broadcast dimensions.
  llvm::SmallVector<int64_t, 4> outShape(shapeLarge.begin(), shapeLarge.end());
  for (const auto& indexPair : llvm::enumerate(broadcastDimensions)) {
    auto newValue = outShapeFiltered[indexPair.index()];
    outShape[indexPair.value()] = newValue;
  }

  return {outShape, elementType};
}

LogicalResult InferBroadcastBinaryOpReturnTypeComponents(
    MLIRContext* context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties,
    std::optional<ArrayRef<int64_t>> broadcastDimensions, Type elementType,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  ShapedType lhsType = cast<ShapedType>(operands[0].getType());
  ShapedType rhsType = cast<ShapedType>(operands[1].getType());
  if (!lhsType || !rhsType ||
      !hlo::isCompatibleElementTypeForHloTypeInference(
          lhsType.getElementType(), rhsType.getElementType()))
    return emitOptionalError(location, "mismatched operand types");
  if (!elementType) elementType = lhsType.getElementType();
  inferredReturnShapes.push_back(
      getBroadcastType(lhsType, rhsType, elementType, broadcastDimensions));
  return success();
}

LogicalResult ReifyBroadcastBinaryOpReturnTypeShapes(
    OpBuilder& builder, Operation* op, ValueRange operands,
    std::optional<ArrayRef<int64_t>> broadcastDimensions,
    SmallVectorImpl<Value>& result) {
  assert(operands.size() == 2 && "expect binary op");
  auto loc = op->getLoc();
  auto lhs = operands[0];
  auto rhs = operands[1];

  // Check for "numpy"-style rank broadcast.
  auto broadcastDimensionsAttr = op->getAttr("broadcast_dimensions");
  if (broadcastDimensions && !hlo::isLegalNumpyRankedBroadcast(
                                 lhs, rhs, broadcastDimensions.value())) {
    // Note: It is unclear whether the general specification of explicit
    // broadcast_dimensions on binary ops is a feature we want to carry
    // forward. While it can technically be implemented for ranked-dynamic,
    // it is incompatible with unranked inputs. If this warning is emitted
    // in real programs, it is an indication that the feature should be
    // implemented versus just falling back on the more standard definition
    // of numpy-like prefix-padding.
    return op->emitWarning()
           << "unsupported non prefix-padded dynamic rank "
           << "broadcast_dimensions = " << broadcastDimensionsAttr;
  }

  result.push_back(hlo::computeBinaryElementwiseBroadcastingResultExtents(
      loc, lhs, rhs, builder));
  return success();
}
}  // namespace

//===----------------------------------------------------------------------===//
// BroadcastComplexOp (has custom type inference due to different result type).
//===----------------------------------------------------------------------===//

LogicalResult BroadcastComplexOp::inferReturnTypeComponents(
    MLIRContext* context, std::optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes,
    OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  ShapedType lhsType = cast<ShapedType>(operands[0].getType());
  Type elementType = ComplexType::get(lhsType.getElementType());
  Adaptor adaptor(operands, attributes, properties, regions);
  return InferBroadcastBinaryOpReturnTypeComponents(
      context, location, operands, attributes, properties,
      adaptor.getBroadcastDimensions(), elementType, inferredReturnShapes);
}
LogicalResult BroadcastComplexOp::reifyReturnTypeShapes(
    OpBuilder& builder, ValueRange operands,
    SmallVectorImpl<Value>& reifiedReturnShapes) {
  return ReifyBroadcastBinaryOpReturnTypeShapes(
      builder, getOperation(), operands, getBroadcastDimensions(),
      reifiedReturnShapes);
}

//===----------------------------------------------------------------------===//
// BroadcastCompareOp (has custom type inference due to different result type).
//===----------------------------------------------------------------------===//

void BroadcastCompareOp::build(OpBuilder& builder, OperationState& result,
                               Value lhs, Value rhs,
                               DenseI64ArrayAttr broadcastDimensions,
                               chlo::ComparisonDirection comparisonDirection,
                               chlo::ComparisonType compareType) {
  build(builder, result, lhs, rhs, broadcastDimensions,
        chlo::ComparisonDirectionAttr::get(builder.getContext(),
                                           comparisonDirection),
        chlo::ComparisonTypeAttr::get(builder.getContext(), compareType));
}

LogicalResult BroadcastCompareOp::inferReturnTypeComponents(
    MLIRContext* context, std::optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes,
    OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  Type elementType = IntegerType::get(context, 1);
  Adaptor adaptor(operands, attributes, properties, regions);
  return InferBroadcastBinaryOpReturnTypeComponents(
      context, location, operands, attributes, properties,
      adaptor.getBroadcastDimensions(), elementType, inferredReturnShapes);
}

LogicalResult BroadcastCompareOp::reifyReturnTypeShapes(
    OpBuilder& builder, ValueRange operands,
    SmallVectorImpl<Value>& reifiedReturnShapes) {
  return ReifyBroadcastBinaryOpReturnTypeShapes(
      builder, getOperation(), operands, getBroadcastDimensions(),
      reifiedReturnShapes);
}

//===----------------------------------------------------------------------===//
// IsInfOp
//===----------------------------------------------------------------------===//

static Type getIsInfLikeReturnType(Value operand) {
  Builder b(operand.getContext());
  return hlo::getSameShapeTensorType(cast<ShapedType>(operand.getType()),
                                     b.getI1Type());
}

LogicalResult IsInfOp::inferReturnTypes(
    MLIRContext* /*ctx*/, std::optional<Location>, ValueRange operands,
    DictionaryAttr, OpaqueProperties, RegionRange,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  inferredReturnTypes.push_back(getIsInfLikeReturnType(operands.front()));
  return success();
}

//===----------------------------------------------------------------------===//
// IsNegInfOp
//===----------------------------------------------------------------------===//

LogicalResult IsNegInfOp::inferReturnTypes(
    MLIRContext* /*ctx*/, std::optional<Location>, ValueRange operands,
    DictionaryAttr, OpaqueProperties, RegionRange,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  inferredReturnTypes.push_back(getIsInfLikeReturnType(operands.front()));
  return success();
}

//===----------------------------------------------------------------------===//
// IsPosInfOp
//===----------------------------------------------------------------------===//

LogicalResult IsPosInfOp::inferReturnTypes(
    MLIRContext* /*ctx*/, std::optional<Location>, ValueRange operands,
    DictionaryAttr, OpaqueProperties, RegionRange,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  inferredReturnTypes.push_back(getIsInfLikeReturnType(operands.front()));
  return success();
}

//===----------------------------------------------------------------------===//
// Macros for method definitions that are common to most broadcasting ops.
//===----------------------------------------------------------------------===//

#define BROADCAST_BINARY_OP_DEFS(Op)                                        \
  LogicalResult Op::inferReturnTypeComponents(                              \
      MLIRContext* context, std::optional<Location> location,               \
      ValueShapeRange operands, DictionaryAttr attributes,                  \
      OpaqueProperties properties, RegionRange regions,                     \
      SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {        \
    Adaptor adaptor(operands.getValues(), attributes, properties, regions); \
    return InferBroadcastBinaryOpReturnTypeComponents(                      \
        context, location, operands, attributes, properties,                \
        adaptor.getBroadcastDimensions(), /*element_type=*/nullptr,         \
        inferredReturnShapes);                                              \
  }                                                                         \
  LogicalResult Op::reifyReturnTypeShapes(                                  \
      OpBuilder& builder, ValueRange operands,                              \
      SmallVectorImpl<Value>& reifiedReturnShapes) {                        \
    return ReifyBroadcastBinaryOpReturnTypeShapes(                          \
        builder, getOperation(), operands, getBroadcastDimensions(),        \
        reifiedReturnShapes);                                               \
  }

BROADCAST_BINARY_OP_DEFS(BroadcastAddOp)
BROADCAST_BINARY_OP_DEFS(BroadcastAndOp)
BROADCAST_BINARY_OP_DEFS(BroadcastAtan2Op)
BROADCAST_BINARY_OP_DEFS(BroadcastDivOp)
BROADCAST_BINARY_OP_DEFS(BroadcastMaxOp)
BROADCAST_BINARY_OP_DEFS(BroadcastMinOp)
BROADCAST_BINARY_OP_DEFS(BroadcastMulOp)
BROADCAST_BINARY_OP_DEFS(BroadcastNextAfterOp)
BROADCAST_BINARY_OP_DEFS(BroadcastOrOp)
BROADCAST_BINARY_OP_DEFS(BroadcastPolygammaOp)
BROADCAST_BINARY_OP_DEFS(BroadcastPowOp)
BROADCAST_BINARY_OP_DEFS(BroadcastRemOp)
BROADCAST_BINARY_OP_DEFS(BroadcastShiftLeftOp)
BROADCAST_BINARY_OP_DEFS(BroadcastShiftRightArithmeticOp)
BROADCAST_BINARY_OP_DEFS(BroadcastShiftRightLogicalOp)
BROADCAST_BINARY_OP_DEFS(BroadcastSubOp)
BROADCAST_BINARY_OP_DEFS(BroadcastXorOp)
BROADCAST_BINARY_OP_DEFS(BroadcastZetaOp)

#undef BROADCAST_BINARY_OP_DEFS

LogicalResult ConstantLikeOp::verify() {
  if (getValue().getType() != getType().getElementType())
    return emitOpError() << "value's type doesn't match element return type";
  return success();
}

LogicalResult ConstantLikeOp::inferReturnTypeComponents(
    MLIRContext* /*context*/, std::optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes,
    OpaqueProperties properties, RegionRange /*regions*/,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  ConstantLikeOp::Adaptor op(operands, attributes, properties);
  if (failed(op.verify(location.value()))) return failure();
  Type elementType = op.getValue().getType();
  Type operandType = op.getOperand().getType();
  if (isa<UnrankedTensorType>(operandType)) {
    inferredReturnShapes.emplace_back(elementType);
  } else {
    const auto& shape = cast<RankedTensorType>(operandType).getShape();
    inferredReturnShapes.emplace_back(shape, elementType);
  }
  return success();
}

LogicalResult ConstantLikeOp::reifyReturnTypeShapes(
    OpBuilder& builder, ValueRange operands,
    SmallVectorImpl<Value>& reifiedReturnShapes) {
  return hlo::deriveShapeFromOperand(&builder, getOperation(), operands.front(),
                                     &reifiedReturnShapes);
}

OpFoldResult ConstantLikeOp::fold(FoldAdaptor /*adaptor*/) {
  auto opType = getOperand().getType();
  if (!opType.hasStaticShape()) return {};
  auto type = RankedTensorType::get(opType.getShape(), getValue().getType());
  if (auto complexAttr = dyn_cast<complex::NumberAttr>(getValue()))
    return DenseElementsAttr::get(type, complexAttr.getValue());
  return DenseElementsAttr::get(type, getValue());
}

LogicalResult BroadcastSelectOp::inferReturnTypeComponents(
    MLIRContext*, std::optional<Location> location, ValueShapeRange operands,
    DictionaryAttr, OpaqueProperties, RegionRange,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  BroadcastSelectOp::Adaptor op(operands.getValues());
  auto predType = cast<ShapedType>(op.getPred().getType());
  auto onTrueType = cast<ShapedType>(op.getOnTrue().getType());
  auto onFalseType = cast<ShapedType>(op.getOnFalse().getType());

  if (onTrueType.getElementType() != onFalseType.getElementType())
    return emitOptionalError(location, "mismatched operand types");

  Type elementType = onTrueType.getElementType();

  // Compute the result shape as two binary broadcasts.
  ShapedTypeComponents& components = inferredReturnShapes.emplace_back(
      getBroadcastType(onTrueType, onFalseType, elementType, std::nullopt));
  if (components.hasRank())
    components = getBroadcastType(
        RankedTensorType::get(components.getDims(), elementType), predType,
        elementType, std::nullopt);
  return success();
}

LogicalResult BroadcastSelectOp::reifyReturnTypeShapes(
    OpBuilder& builder, ValueRange operands, SmallVectorImpl<Value>& result) {
  result.push_back(hlo::computeNaryElementwiseBroadcastingResultExtents(
      getLoc(), operands, builder));
  return success();
}

//===----------------------------------------------------------------------===//
// RaggedDotOp
//===----------------------------------------------------------------------===//

namespace {

// RaggedDot has three general modes, based on the kind of the ragged dimension.
// Mode 1, where the ragged dimension is an lhs non-contracting dim (m).
//   lhs : [b, m, k]
//   rhs : [g, b, k, n]
//   group_sizes : [b, g]
//   result : [b, m, n]
// Mode 2, where the ragged dimension is an lhs/rhs contracting dim (k).
//   lhs : [b, m, k]
//   rhs : [b, k, n]
//   group_sizes : [b, g]
//   result : [g, b, m, n]
// Mode 3, where the ragged dimension is an lhs/rhs batch dim (b).
//   lhs : [b, m, k]
//   rhs : [b, k, n]
//   group_sizes : [g]
//   result : [b, m, n]
// As with dot_general, the lhs and rhs can have arbitrary batching,
// contracting and non-contracting dimensions.
// The group_sizes arg has the shape [b...,x...,g], where:
// - b... are all the lhs batch dims before (outer-to) the lhs ragged dim,
// - x... are,
//   - in mode 1, all the lhs non-contracting dims before the lhs ragged dim,
//   - in mode 2, all the lhs contracting dims before the lhs ragged dim, and
//   - in mode 3, empty;
// - g is the number of groups in the lhs ragged dim.
// Additionally:
//   - In all modes, the lhs must have exactly one ragged dimension.
//   - In mode 1, the rhs must have exactly one group dimension.
//   - If a group_sizes of shape [g] is passed, it is broadcasted according to
//     the rules above.
LogicalResult checkRaggedDotConstraints(
    std::optional<Location> location, RankedTensorType rankedLhsType,
    RankedTensorType rankedRhsType, RankedTensorType rankedGroupSizesType,
    ArrayRef<int64_t> lhsBatchingDimensions,
    ArrayRef<int64_t> rhsBatchingDimensions,
    ArrayRef<int64_t> lhsContractingDimensions,
    ArrayRef<int64_t> rhsContractingDimensions,
    ArrayRef<int64_t> lhsRaggedDimensions,
    ArrayRef<int64_t> rhsGroupDimensions) {
  // Check that there is exactly one lhs ragged dimension.
  if (lhsRaggedDimensions.size() != 1) {
    return emitOptionalError(
        location, "There must be exactly one ragged dimension in the lhs.");
  }
  const int64_t lhsRaggedDim = lhsRaggedDimensions[0];

  // Check that the lhs ragged dimension is in range.
  if (failed(hlo::checkDimInBounds(location, lhsRaggedDim,
                                   rankedLhsType.getRank(), "lhs_ragged_dim",
                                   "lhs_rank"))) {
    return failure();
  }

  enum Mode {
    // Ragged non-contracting (m): [b,m,k], [g,b,k,n], [b,g] -> [b,m,n].
    kNonContracting,
    // Ragged contracting (k):     [b,m,k], [b,k,n],   [b,g] -> [g,b,m,n].
    kContracting,
    // Ragged batch (b):           [b,m,k], [b,k,n],   [g]   -> [b,m,n].
    kBatch
  };
  Mode mode;
  if (llvm::is_contained(lhsBatchingDimensions, lhsRaggedDim)) {
    mode = kBatch;
  } else if (llvm::is_contained(lhsContractingDimensions, lhsRaggedDim)) {
    mode = kContracting;
  } else {
    mode = kNonContracting;
  }

  // Validate the shape of group_sizes.
  {
    // Construct the expected shape [b...,x...,g] of group_sizes.
    SmallVector<int64_t> prefixDims;
    prefixDims.reserve(rankedLhsType.getRank() - 1);
    prefixDims.insert(prefixDims.end(), lhsBatchingDimensions.begin(),
                      lhsBatchingDimensions.end());
    switch (mode) {
      case kBatch:
        prefixDims.resize(
            std::distance(lhsBatchingDimensions.begin(),
                          llvm::find(lhsBatchingDimensions, lhsRaggedDim)));
        break;
      case kContracting:
        prefixDims.insert(prefixDims.end(), lhsContractingDimensions.begin(),
                          llvm::find(lhsContractingDimensions, lhsRaggedDim));
        break;
      case kNonContracting:
        for (int64_t i = 0; i < lhsRaggedDim; ++i) {
          if (!llvm::is_contained(lhsBatchingDimensions, i) &&
              !llvm::is_contained(lhsContractingDimensions, i)) {
            prefixDims.push_back(i);
          }
        }
        break;
    }
    SmallVector<int64_t> expectedPrefix;
    expectedPrefix.reserve(prefixDims.size());
    for (const int64_t dim : prefixDims) {
      expectedPrefix.push_back(rankedLhsType.getDimSize(dim));
    }

    // Validate the actual shape, if it was passed as something other than [g].
    if (rankedGroupSizesType.getRank() != 1) {
      if (rankedGroupSizesType.getRank() !=
          static_cast<int64_t>(expectedPrefix.size()) + 1) {
        return emitOptionalError(location, "expected group_sizes to have rank ",
                                 expectedPrefix.size() + 1, ", got ",
                                 rankedGroupSizesType.getRank());
      }
      auto groupSizesShape = rankedGroupSizesType.getShape();
      if (!std::equal(expectedPrefix.begin(), expectedPrefix.end(),
                      groupSizesShape.begin())) {
        auto nonEmptyShapeStr = [](ArrayRef<int64_t> shape) {
          std::string s = "";
          for (size_t i = 0; i < shape.size() - 1; ++i) {
            s += std::to_string(shape[i]) + ", ";
          }
          return s + std::to_string(shape.back());
        };
        return emitOptionalError(
            location, "group_sizes is expected to have shape [",
            nonEmptyShapeStr(expectedPrefix), ", ", groupSizesShape.back(),
            "], got [", nonEmptyShapeStr(groupSizesShape), "]");
      }
    }
  }
  const int64_t numGroups = rankedGroupSizesType.getShape().back();

  // Validate basic properties of the rhs group dimension(s).
  for (auto rhsGroupDim : rhsGroupDimensions) {
    if (failed(hlo::checkDimInBounds(location, rhsGroupDim,
                                     rankedRhsType.getRank(), "rhs_group_dim",
                                     "rhs_rank"))) {
      return failure();
    }
  }
  if (failed(hlo::checkDimsDistinct(
          location, rhsGroupDimensions, rhsBatchingDimensions,
          "rhs_group_dimensions", "rhs_batching_dimensions")) ||
      failed(hlo::checkDimsDistinct(
          location, rhsGroupDimensions, rhsContractingDimensions,
          "rhs_group_dimensions", "rhs_contracting_dimensions"))) {
    return failure();
  }

  switch (mode) {
    case kBatch:
      [[fallthrough]];
    case kContracting:
      if (!rhsGroupDimensions.empty()) {
        return emitOptionalError(
            location,
            "There must be zero group dimensions in the rhs when the "
            "ragged dimension is batch or contracting.");
      }
      break;
    case kNonContracting:
      if (rhsGroupDimensions.size() != 1) {
        return emitOptionalError(
            location,
            "There must be exactly one group dimension in the rhs when the lhs "
            "ragged dimension is non-contracting.");
      }
      // Compare the group dimension size with the number of groups.
      const int64_t rhsGroupDim = rhsGroupDimensions[0];
      if (!hlo::verifyCompatibleDims(numGroups,
                                     rankedRhsType.getDimSize(rhsGroupDim))) {
        return emitOptionalError(
            location,
            "rhs group dimension is expected to have size=", numGroups,
            ", got ", rankedRhsType.getDimSize(rhsGroupDim));
      }
      break;
  }
  return success();
}

SmallVector<int64_t> inferRaggedDotOutputDimensions(
    RankedTensorType rankedLhsType, RankedTensorType rankedRhsType,
    RankedTensorType rankedGroupSizesType,
    ArrayRef<int64_t> lhsBatchingDimensions,
    ArrayRef<int64_t> rhsBatchingDimensions,
    ArrayRef<int64_t> lhsContractingDimensions,
    ArrayRef<int64_t> rhsContractingDimensions,
    ArrayRef<int64_t> lhsRaggedDimensions,
    ArrayRef<int64_t> rhsGroupDimensions) {
  // Must have already checked that there is exactly one lhs ragged dim.
  const int64_t lhsRaggedDim = lhsRaggedDimensions[0];
  // Must have already checked the shape of group_sizes.
  const int64_t numGroups = rankedGroupSizesType.getShape().back();

  SmallVector<int64_t> dimensions;
  // Add the group dimension to the result shape in case of ragged contracting.
  if (llvm::is_contained(lhsContractingDimensions, lhsRaggedDim)) {
    dimensions.push_back(numGroups);
  }
  auto lhsShape = rankedLhsType.getShape();
  auto rhsShape = rankedRhsType.getShape();
  for (const int64_t lhsBatchingDim : lhsBatchingDimensions)
    dimensions.push_back(lhsShape[lhsBatchingDim]);
  for (int64_t i = 0; i < rankedLhsType.getRank(); i++)
    if (!llvm::is_contained(lhsBatchingDimensions, i) &&
        !llvm::is_contained(lhsContractingDimensions, i))
      dimensions.push_back(lhsShape[i]);
  for (int64_t i = 0; i < rankedRhsType.getRank(); i++)
    if (!llvm::is_contained(rhsBatchingDimensions, i) &&
        !llvm::is_contained(rhsContractingDimensions, i) &&
        !llvm::is_contained(rhsGroupDimensions, i))
      dimensions.push_back(rhsShape[i]);
  return dimensions;
}

LogicalResult inferRaggedDotOp(
    std::optional<Location> location, Value lhs, Value rhs, Value groupSizes,
    ArrayRef<int64_t> lhsBatchingDimensions,
    ArrayRef<int64_t> rhsBatchingDimensions,
    ArrayRef<int64_t> lhsContractingDimensions,
    ArrayRef<int64_t> rhsContractingDimensions,
    ArrayRef<int64_t> lhsRaggedDimensions, ArrayRef<int64_t> rhsGroupDimensions,
    std::optional<ArrayAttr> precisionConfig,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  if (failed(hlo::verifyPrecisionConfig(location, precisionConfig))) {
    return failure();
  }

  // Validate basic properties of dot dimension numbers.
  if (failed(hlo::checkDotGeneralConstraints(
          location, lhs.getType(), rhs.getType(), lhsBatchingDimensions,
          rhsBatchingDimensions, lhsContractingDimensions,
          rhsContractingDimensions, precisionConfig))) {
    return failure();
  }

  // Validate ragged dot constraints.
  auto rankedLhsType = cast<RankedTensorType>(lhs.getType());
  auto rankedRhsType = cast<RankedTensorType>(rhs.getType());
  auto rankedGroupSizesType = cast<RankedTensorType>(groupSizes.getType());
  if (failed(checkRaggedDotConstraints(
          location, rankedLhsType, rankedRhsType, rankedGroupSizesType,
          lhsBatchingDimensions, rhsBatchingDimensions,
          lhsContractingDimensions, rhsContractingDimensions,
          lhsRaggedDimensions, rhsGroupDimensions))) {
    return failure();
  }

  // Infer the output dimensions of the ragged dot operation.
  inferredReturnShapes.emplace_back(inferRaggedDotOutputDimensions(
      rankedLhsType, rankedRhsType, rankedGroupSizesType, lhsBatchingDimensions,
      rhsBatchingDimensions, lhsContractingDimensions, rhsContractingDimensions,
      lhsRaggedDimensions, rhsGroupDimensions));
  return success();
}

}  // namespace

LogicalResult RaggedDotOp::verify() {
  auto location = getLoc();
  auto raggedDotDimNums = getRaggedDotDimensionNumbers();

  SmallVector<ShapedTypeComponents> inferredReturnShapes;
  if (failed(inferRaggedDotOp(location, getLhs(), getRhs(), getGroupSizes(),
                              raggedDotDimNums.getLhsBatchingDimensions(),
                              raggedDotDimNums.getRhsBatchingDimensions(),
                              raggedDotDimNums.getLhsContractingDimensions(),
                              raggedDotDimNums.getRhsContractingDimensions(),
                              raggedDotDimNums.getLhsRaggedDimensions(),
                              raggedDotDimNums.getRhsGroupDimensions(),
                              getPrecisionConfig(), inferredReturnShapes)))
    return failure();
  auto inferredShape = inferredReturnShapes[0];

  auto resultType = cast<ShapedType>(getResult().getType());
  if (failed(verifyCompatibleShape(inferredShape.getDims(),
                                   resultType.getShape()))) {
    return emitOptionalError(
        location, "inferred shape '",
        hlo::dimSizesToString(inferredShape.getDims()), "' ",
        "is incompatible with return type of operation ", resultType, "");
  }

  return success();
}

LogicalResult RaggedDotOp::inferReturnTypes(
    MLIRContext*, std::optional<Location>, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  RaggedDotOp::Adaptor op(operands, attributes, properties, regions);

  auto rankedLhsType = cast<RankedTensorType>(op.getLhs().getType());
  auto rankedRhsType = cast<RankedTensorType>(op.getRhs().getType());
  auto rankedGroupSizesType =
      cast<RankedTensorType>(op.getGroupSizes().getType());
  auto raggedDotDimNums = op.getRaggedDotDimensionNumbers();

  inferredReturnTypes.push_back(RankedTensorType::get(
      inferRaggedDotOutputDimensions(
          rankedLhsType, rankedRhsType, rankedGroupSizesType,
          raggedDotDimNums.getLhsBatchingDimensions(),
          raggedDotDimNums.getRhsBatchingDimensions(),
          raggedDotDimNums.getLhsContractingDimensions(),
          raggedDotDimNums.getRhsContractingDimensions(),
          raggedDotDimNums.getLhsRaggedDimensions(),
          raggedDotDimNums.getRhsGroupDimensions()),
      rankedLhsType.getElementType()));
  return success();
}

//===----------------------------------------------------------------------===//
// TopKOp
//===----------------------------------------------------------------------===//

LogicalResult TopKOp::inferReturnTypeComponents(
    MLIRContext* context, std::optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes,
    OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  TopKOp::Adaptor adaptor(operands, attributes, properties, regions);
  return hlo::inferTopKOp(location, adaptor.getOperand(), adaptor.getK(),
                          inferredReturnShapes);
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

OpFoldResult ConstantOp::fold(FoldAdaptor /*adaptor*/) { return getValue(); }

LogicalResult ConstantOp::inferReturnTypes(
    MLIRContext*, std::optional<Location>, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  Adaptor adaptor(operands, attributes, properties, regions);
  Type type = cast<TypedAttr>(adaptor.getValueAttr()).getType();
  inferredReturnTypes.push_back(type);
  return success();
}

}  // namespace chlo
}  // namespace mlir

#define GET_OP_CLASSES
#include "stablehlo/dialect/ChloOps.cpp.inc"

namespace mlir {
namespace chlo {

namespace {
struct ChloDialectInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  // Allow all call operations to be inlined.
  bool isLegalToInline(Operation* call, Operation* callable,
                       bool wouldBeCloned) const final {
    return true;
  }
  // We don't have any special restrictions on what can be inlined into
  // destination regions (e.g. while/conditional bodies). Always allow it.
  bool isLegalToInline(Region* dest, Region* src, bool wouldBeCloned,
                       IRMapping& valueMapping) const final {
    return true;
  }
  // Operations in CHLO dialect are always legal to inline since they are pure.
  bool isLegalToInline(Operation*, Region*, bool, IRMapping&) const final {
    return true;
  }
};
}  // namespace

//===----------------------------------------------------------------------===//
// chlo Dialect Constructor
//===----------------------------------------------------------------------===//

ChloDialect::ChloDialect(MLIRContext* context)
    : Dialect(getDialectNamespace(), context, TypeID::get<ChloDialect>()) {
  addOperations<
#define GET_OP_LIST
#include "stablehlo/dialect/ChloOps.cpp.inc"
      >();
  addInterfaces<ChloDialectInlinerInterface>();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "stablehlo/dialect/ChloAttrs.cpp.inc"
      >();

  addBytecodeInterface(this);
}

Operation* ChloDialect::materializeConstant(OpBuilder& builder, Attribute value,
                                            Type type, Location loc) {
  if (isa<ElementsAttr>(value))
    return builder.create<chlo::ConstantOp>(loc, type,
                                            cast<ElementsAttr>(value));
  return nullptr;
}

// Entry point for Attribute parsing, TableGen generated code will handle the
// dispatch to the individual classes.
Attribute ChloDialect::parseAttribute(DialectAsmParser& parser,
                                      Type type) const {
  StringRef attrTag;
  Attribute attr;
  auto parseResult = generatedAttributeParser(parser, &attrTag, type, attr);
  if (parseResult.has_value()) return attr;
  parser.emitError(parser.getNameLoc(), "unknown chlo attribute");
  return Attribute();
}

// Entry point for Attribute printing, TableGen generated code will handle the
// dispatch to the individual classes.
void ChloDialect::printAttribute(Attribute attr, DialectAsmPrinter& os) const {
  LogicalResult result = generatedAttributePrinter(attr, os);
  (void)result;
  assert(succeeded(result));
}

/// Helpers for attributes parsing.

static ParseResult parseDims(AsmParser& parser,
                             SmallVector<int64_t>& dimSizes) {
  dimSizes.clear();
  auto failOrDims = hlo::parseDimSizes(parser);
  if (failed(failOrDims)) return failure();
  dimSizes = std::move(*failOrDims);
  return success();
}

/// Parse a custom attribute that resembles a struct of the form
/// <
///   foo = something_parsed_by_custom_parser,
///   bar = something_parsed_by_different_custom_parser,
///   baz something_parsed_by_another_custom_parser
/// >
/// The optional argument `parse_equal` array can be used to denote if
/// '=' follows the keyword (see baz in the example above) for a field. If
/// not provided, all fields must be followed by a '='.
static ParseResult parseStruct(
    AsmParser& parser, ArrayRef<StringRef> keywords,
    ArrayRef<llvm::function_ref<ParseResult()>> parseFuncs,
    ArrayRef<bool> parseEqual = {}) {
  assert(keywords.size() == parseFuncs.size());
  assert(parseEqual.empty() || parseEqual.size() == keywords.size());
  SmallVector<bool> seen(keywords.size(), false);
  while (failed(parser.parseOptionalGreater())) {
    bool foundOne = false;
    for (const auto& it : llvm::enumerate(keywords)) {
      size_t index = it.index();
      StringRef keyword = it.value();
      if (failed(parser.parseOptionalKeyword(keyword))) continue;
      if (seen[index])
        return parser.emitError(parser.getCurrentLocation())
               << "duplicated `" << keyword << "` entry";
      if (parseEqual.empty() || parseEqual[index]) {
        if (failed(parser.parseEqual())) return failure();
      }
      if (failed(parseFuncs[index]())) return failure();
      if (failed(parser.parseOptionalComma())) return parser.parseGreater();
      seen[index] = true;
      foundOne = true;
    }
    if (!foundOne) {
      auto parseError = parser.emitError(parser.getCurrentLocation())
                        << "expected one of: ";
      llvm::interleaveComma(keywords, parseError, [&](StringRef kw) {
        parseError << '`' << kw << '`';
      });
      return parseError;
    }
  }
  return success();
}

// Helpers to print an optional array or integer field, to simplify writing
// attribute printers.
template <typename T>
static void printField(AsmPrinter& printer, StringRef name, T field,
                       StringRef& separator) {
  if (field != 0) {
    printer << separator << name << " = " << field;
    separator = ", ";
  }
}
template <typename T>
static void printField(AsmPrinter& printer, StringRef name, ArrayRef<T> field,
                       StringRef& separator) {
  if (!field.empty()) {
    printer << separator << name << " = [";
    llvm::interleaveComma(field, printer);
    printer << "]";
    separator = ", ";
  }
}
template <typename... Ts>
static void printStruct(AsmPrinter& printer, StringRef name,
                        Ts... printFields) {
  printer << "<";
  StringRef separator = "";
  // Fold expression to print each entry in the parameter pack.
  // TODO(stablehlo-team): this can be simplified when TF moves to C++17.
  using unused = int[];
  (void)unused{0, (printField(printer, std::get<0>(printFields),
                              std::get<1>(printFields), separator),
                   0)...};
  printer << ">";
}

// Custom printer and parser for RaggedDotDimensionNumbersAttr.
void RaggedDotDimensionNumbersAttr::print(AsmPrinter& printer) const {
  printStruct(
      printer, "ragged_dot",
      std::make_pair("lhs_batching_dimensions", getLhsBatchingDimensions()),
      std::make_pair("rhs_batching_dimensions", getRhsBatchingDimensions()),
      std::make_pair("lhs_contracting_dimensions",
                     getLhsContractingDimensions()),
      std::make_pair("rhs_contracting_dimensions",
                     getRhsContractingDimensions()),
      std::make_pair("lhs_ragged_dimensions", getLhsRaggedDimensions()),
      std::make_pair("rhs_group_dimensions", getRhsGroupDimensions()));
}

Attribute RaggedDotDimensionNumbersAttr::parse(AsmParser& parser, Type type) {
  if (failed(parser.parseLess())) return {};

  SmallVector<int64_t> lhsBatchingDimensions;
  SmallVector<int64_t> rhsBatchingDimensions;
  SmallVector<int64_t> lhsContractingDimensions;
  SmallVector<int64_t> rhsContractingDimensions;
  SmallVector<int64_t> lhsRaggedDimensions;
  SmallVector<int64_t> rhsGroupDimensions;

  if (failed(parseStruct(
          parser,
          {"lhs_batching_dimensions", "rhs_batching_dimensions",
           "lhs_contracting_dimensions", "rhs_contracting_dimensions",
           "lhs_ragged_dimensions", "rhs_group_dimensions"},
          {[&]() { return parseDims(parser, lhsBatchingDimensions); },
           [&]() { return parseDims(parser, rhsBatchingDimensions); },
           [&]() { return parseDims(parser, lhsContractingDimensions); },
           [&]() { return parseDims(parser, rhsContractingDimensions); },
           [&]() { return parseDims(parser, lhsRaggedDimensions); },
           [&]() { return parseDims(parser, rhsGroupDimensions); }}))) {
    parser.emitError(parser.getCurrentLocation())
        << "failed parsing ragged dot dimension numbers attribute";
    return {};
  }
  return RaggedDotDimensionNumbersAttr::get(
      parser.getContext(), lhsBatchingDimensions, rhsBatchingDimensions,
      lhsContractingDimensions, rhsContractingDimensions, lhsRaggedDimensions,
      rhsGroupDimensions);
}

}  // namespace chlo
}  // namespace mlir
