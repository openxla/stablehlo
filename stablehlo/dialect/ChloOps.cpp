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

#include <cassert>
#include <cstdint>
#include <optional>

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

}  // namespace chlo
}  // namespace mlir
