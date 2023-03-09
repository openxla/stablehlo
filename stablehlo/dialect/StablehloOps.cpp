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

#include "stablehlo/dialect/StablehloOps.h"

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <numeric>
#include <optional>
#include <set>
#include <unordered_map>
#include <utility>

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/Regex.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/InliningUtils.h"
#include "stablehlo/dialect/AssemblyFormat.h"
#include "stablehlo/dialect/StablehloBytecode.h"
#include "stablehlo/dialect/StablehloOps.h.inc"
#include "stablehlo/dialect/TypeInference.h"

// Include order matters
using mlir::hlo::parseDimSizes;
using mlir::hlo::printDimSizes;
#include "stablehlo/dialect/StablehloEnums.cpp.inc"
#define GET_ATTRDEF_CLASSES
#include "stablehlo/dialect/StablehloAttrs.cpp.inc"

namespace mlir {
namespace stablehlo {
namespace {

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

hlo::HloDialectInterface* getStablehloDialect(MLIRContext* context) {
  StablehloDialect* dialect = context->getLoadedDialect<StablehloDialect>();
  return dialect->getRegisteredInterface<hlo::HloDialectInterface>();
}

void createArgs(ArrayRef<OpAsmParser::UnresolvedOperand> operands,
                ArrayRef<Type> types,
                SmallVector<OpAsmParser::Argument>& args) {
  for (auto argAndType : llvm::zip(operands, types)) {
    auto& arg = args.emplace_back();
    arg.ssaName = std::get<0>(argAndType);
    arg.type = std::get<1>(argAndType);
  }
}

// Returns a new scalar integer value having type `type`. Here `type` must be
// an integer or index type.
Value maybeCastTo(OpBuilder& b, Location loc, Value value, Type type) {
  if (type == value.getType()) return value;
  assert(type.isIndex() || value.getType().isIndex());
  return b.create<arith::IndexCastOp>(loc, type, value);
}
}  // namespace

LogicalResult TypeExtensionsAttr::verifyEncoding(
    llvm::ArrayRef<int64_t> shape, mlir::Type elementType,
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError) const {
  return hlo::verifyBounds(
      getBounds(), RankedTensorType::get(shape, elementType), emitError);
}

//===----------------------------------------------------------------------===//
// CollectivePermuteOp
//===----------------------------------------------------------------------===//

void CollectivePermuteOp::build(OpBuilder& odsBuilder, OperationState& odsState,
                                Type resultType, Value operand,
                                DenseIntElementsAttr sourceTargetPairs) {
  CollectivePermuteOp::build(odsBuilder, odsState, resultType, operand,
                             sourceTargetPairs, /*channel_handle=*/nullptr);
}

//===----------------------------------------------------------------------===//
// ReduceScatterOp
//===----------------------------------------------------------------------===//

LogicalResult ReduceScatterOp::verify() {
  return hlo::verifyReduceScatterOp(
      getLoc(), getOperand(), getScatterDimension(), getReplicaGroups(),
      getUseGlobalDeviceIds(), getComputation(), getResult());
}

//===----------------------------------------------------------------------===//
// CompatibleOperandsAndResultType
//===----------------------------------------------------------------------===//

// TODO(b/231358795): Review the use of InferTypeOpInterface for ops that
// support quantization or sparsity.
#define INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(Op)                        \
  LogicalResult Op::inferReturnTypeComponents(                                \
      MLIRContext* context, std::optional<Location> location,                 \
      ValueShapeRange operands, DictionaryAttr attributes,                    \
      RegionRange regions,                                                    \
      SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {          \
    return inferReturnTypeComponentsFromOperands(context, location, operands, \
                                                 attributes, regions,         \
                                                 inferredReturnShapes);       \
  }

INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(AddOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(AllReduceOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(AndOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(Atan2Op)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(CbrtOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(CeilOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(ClzOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(CollectivePermuteOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(CosineOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(CrossReplicaSumOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(DivOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(ExpOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(Expm1Op)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(FloorOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(LogOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(Log1pOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(LogisticOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(MaxOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(MinOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(MulOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(NegOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(NotOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(OrOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(PopulationCountOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(PowOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(ReducePrecisionOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(RemOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(RoundNearestEvenOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(RoundOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(RsqrtOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(ShiftLeftOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(ShiftRightArithmeticOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(ShiftRightLogicalOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(SignOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(SineOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(SqrtOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(SubtractOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(TanhOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(XorOp)

//===----------------------------------------------------------------------===//
// AfterAllOp
//===----------------------------------------------------------------------===//

LogicalResult AfterAllOp::inferReturnTypes(
    MLIRContext* context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  return hlo::inferAfterAllOp(getStablehloDialect(context), location,
                              inferredReturnTypes);
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

OpFoldResult ConstantOp::fold(FoldAdaptor adaptor) {
  assert(adaptor.getOperands().empty() && "constant has no operands");

  // Return the held attribute value.
  return getValue();
}

// Builds a constant op with the specified attribute `value`.
void ConstantOp::build(OpBuilder& /*builder*/, OperationState& result,
                       Attribute value) {
  Type type;
  if (auto elemAttr = value.dyn_cast<ElementsAttr>()) {
    type = elemAttr.getType();
  } else if (value.isa<BoolAttr, FloatAttr, IntegerAttr>()) {
    // All XLA types must be tensor types. In the build() method, we want to
    // provide more flexibility by allowing attributes of scalar types. But we
    // need to wrap it up with ElementsAttr to construct valid XLA constants.
    type =
        RankedTensorType::get(/*shape=*/{}, value.cast<TypedAttr>().getType());
    value = DenseElementsAttr::get(type.cast<TensorType>(), value);
  } else if (auto complexAttr = value.dyn_cast<complex::NumberAttr>()) {
    type = RankedTensorType::get(/*shape=*/{},
                                 complexAttr.cast<TypedAttr>().getType());
    value =
        DenseElementsAttr::get(type.cast<TensorType>(), complexAttr.getValue());
  }

  // TODO: support other XLA specific types.
  assert(type && "unsupported attribute type for building constant");
  result.types.push_back(type);
  result.addAttribute("value", value);
}

LogicalResult ConstantOp::inferReturnTypes(
    MLIRContext*, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  ConstantOpAdaptor adaptor(operands, attributes);
  return hlo::inferConstantOp(location, adaptor.getValue(),
                              inferredReturnTypes);
}

bool ConstantOp::isCompatibleReturnTypes(TypeRange l, TypeRange r) {
  if (l.size() != r.size() || l.size() != 1) return false;
  auto lhsTy = l.front().cast<TensorType>();
  auto rhsTy = r.front().cast<TensorType>();
  // For comparisons of the uniform quantized element based tensor type, use the
  // storage type since the constant value will be stored through the underlying
  // storage type.
  if (auto rhsElemTy = rhsTy.getElementType().dyn_cast<quant::QuantizedType>())
    rhsTy = hlo::getSameShapeTensorType(rhsTy, rhsElemTy.getStorageType());
  return lhsTy == rhsTy;
}

ParseResult ConstantOp::parse(OpAsmParser& parser, OperationState& result) {
  // Parse the generic form.
  if (succeeded(parser.parseOptionalLParen())) {
    if (parser.parseRParen()) return failure();
    if (parser.parseOptionalAttrDict(result.attributes)) return failure();
    if (parser.parseColon() || parser.parseLParen() || parser.parseRParen() ||
        parser.parseArrow())
      return failure();
    Type resultTy;
    if (parser.parseType(resultTy)) return failure();
    result.addTypes(resultTy);
    return success();
  }

  ElementsAttr valueAttr;
  if (parser.parseOptionalAttrDict(result.attributes)) return failure();
  if (parser.parseCustomAttributeWithFallback(valueAttr, Type{}, "value",
                                              result.attributes))
    return failure();
  result.addTypes(valueAttr.getType());
  return success();
}

/// Print a `constant` op.
///
/// op ::= attr-dict $value
///
/// When the `value` and `output` have different type, it just uses the default
/// operator assembly format as a fallback.
void ConstantOp::print(::mlir::OpAsmPrinter& p) {
  // If not all types are the same, use generic form.
  if (getValue().getType() != getType()) {
    p.printGenericOp(getOperation(), /*printOpName=*/false);
    return;
  }

  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"value"});
  p << ' ';
  p.printStrippedAttrOrType(getValueAttr());
}

//===----------------------------------------------------------------------===//
// CreateTokenOp
//===----------------------------------------------------------------------===//

LogicalResult CreateTokenOp::inferReturnTypes(
    MLIRContext* context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  return hlo::inferCreateTokenOp(getStablehloDialect(context), location,
                                 inferredReturnTypes);
}

//===----------------------------------------------------------------------===//
// CustomCallOp
//===----------------------------------------------------------------------===//

LogicalResult CustomCallOp::verify() {
  // If both operand and result layout attributes are not specified then nothing
  // to verify.
  if (getOperandLayouts().has_value() || getResultLayouts().has_value()) {
    // Layout constraints for either both operands & results or none should be
    // specified.
    if (getOperandLayouts().has_value() != getResultLayouts().has_value())
      return emitOpError() << "Layout attributes should be specified for "
                              "either both operands and results or none.";

    // Helper function to verify types and the corresponding layouts.
    auto verifyTypesAndLayouts =
        [this](TypeRange types, mlir::ArrayAttr layouts,
               const std::string& valueName) -> LogicalResult {
      if (types.size() != layouts.size())
        return emitOpError()
               << "Number of " << valueName << "s must match the number of "
               << valueName << " layouts, " << types.size()
               << " != " << layouts.size();

      for (const auto& indexedTypeAndLayout :
           llvm::enumerate(llvm::zip(types, layouts))) {
        // Get index for more descriptive error message.
        auto index = indexedTypeAndLayout.index();

        auto type = std::get<0>(indexedTypeAndLayout.value());
        auto layout = std::get<1>(indexedTypeAndLayout.value())
                          .cast<DenseIntElementsAttr>();

        if (type.isa<TupleType>())
          return emitOpError() << "Tuple types are not fully supported with "
                                  "layout constraints yet";
        auto tensorType = type.dyn_cast<TensorType>();

        // For non-tensor types such as !stablehlo.token, the layout should be
        // empty.
        if (!tensorType) {
          if (layout.empty()) continue;
          return emitOpError()
                 << "Only tensor types can have non-empty layout: " << valueName
                 << " #" << index << " of type " << type << " has layout "
                 << layout;
        }

        // For unranked tensors, we cannot verify the compatibility with layout
        // any further.
        if (!tensorType.hasRank()) continue;

        // Layout must be a permutation of [0, N) where N is the rank of the
        // tensor type.
        std::vector<int64_t> range(tensorType.getRank());
        std::iota(range.begin(), range.end(), 0);
        if (tensorType.getRank() != layout.size() ||
            !std::is_permutation(range.begin(), range.end(), layout.begin()))
          return emitOpError()
                 << "incorrect layout " << layout << " for type " << type
                 << ", layout must be a permutation of [0, "
                 << tensorType.getRank() << ")";
      }
      return success();
    };

    // At this point both `operand_layouts` and `result_layouts` are defined.
    ArrayAttr operandLayouts = this->getOperandLayouts().value();
    ArrayAttr resultLayouts = this->getResultLayouts().value();

    // Full support for layouts for arbitrary nesting of tuples is not
    // supported yet.
    //
    // If result does not have any tuples, then i-th element of `result_layouts`
    // specifies the layout constraints on i-th result.
    //
    // For the common case of a single tuple result packing non-tuple values,
    // the i-th element of `result_layouts` specifies layout for i-th element of
    // the result tuple.
    TypeRange resultTypes;
    if (getNumResults() == 1 && getResult(0).getType().isa<TupleType>())
      resultTypes = getResult(0).getType().cast<TupleType>().getTypes();
    else
      resultTypes = getResultTypes();

    // Verify that operands and operand layouts match.
    if (failed(verifyTypesAndLayouts(getOperandTypes(), operandLayouts,
                                     "operand")))
      return failure();

    // Verify that results and result layouts match.
    if (failed(verifyTypesAndLayouts(resultTypes, resultLayouts, "result")))
      return failure();
  }

  // Check output_operand_aliases

  auto aliasArrayAttr = getOutputOperandAliases();
  for (auto attr : aliasArrayAttr) {
    auto alias = attr.cast<OutputOperandAliasAttr>();
    auto outputTupleIndices = alias.getOutputTupleIndices();
    auto operandIndex = alias.getOperandIndex();
    auto operandTupleIndices = alias.getOperandTupleIndices();

    if (operandIndex < 0 ||
        operandIndex >= static_cast<int64_t>(getInputs().size()))
      return emitOpError()
             << "expects operandIndex in the output_operand_alias attribute "
                "to be in range [0, "
             << getInputs().size() << "); got: " << operandIndex << ".";

    Type operandPart = getOperand(operandIndex).getType();
    for (auto i : operandTupleIndices) {
      if (!operandPart.isa<TupleType>() ||
          i >= static_cast<int64_t>(operandPart.cast<TupleType>().size()) ||
          i < 0)
        return emitOpError()
               << "operand_tuple_indices in the output_operand_alias "
                  "attribute out of bounds";
      operandPart = operandPart.cast<TupleType>().getType(i);
    }
    Type outputPart = getNumResults() > 1
                          ? TupleType::get(getContext(), getResultTypes())
                          : getResult(0).getType();
    for (auto i : outputTupleIndices) {
      if (!outputPart.isa<TupleType>() ||
          i >= static_cast<int64_t>(outputPart.cast<TupleType>().size()) ||
          i < 0)
        return emitOpError()
               << "output_tuple_indices in the output_operand_alias "
                  "attribute out of bounds";
      outputPart = outputPart.cast<TupleType>().getType(i);
    }
    if (operandPart != outputPart)
      return emitOpError()
             << "shapes mismatch in the output_operand_alias attribute: "
             << "operand part has type " << operandPart
             << " and output part has type " << outputPart;
  }
  return success();
}

void CustomCallOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>&
        effects) {
  // CustomCall has "all possible effects" unless the has_side_effect is present
  // and set to false.
  auto hasSideEffect = (*this)->getAttrOfType<BoolAttr>("has_side_effect");
  if (hasSideEffect && !hasSideEffect.getValue()) return;
  effects.emplace_back(MemoryEffects::Allocate::get());
  effects.emplace_back(MemoryEffects::Free::get());
  effects.emplace_back(MemoryEffects::Write::get());
  effects.emplace_back(MemoryEffects::Read::get());
}

//===----------------------------------------------------------------------===//
// CholeskyOp
//===----------------------------------------------------------------------===//

LogicalResult CholeskyOp::inferReturnTypeComponents(
    MLIRContext*, std::optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  CholeskyOp::Adaptor adaptor(operands, attributes, regions);
  return hlo::inferCholeskyOp(location, adaptor.getA(), inferredReturnShapes);
}

//===----------------------------------------------------------------------===//
// DotOp
//===----------------------------------------------------------------------===//

LogicalResult DotOp::verify() {
  return hlo::verifyDotOp(getLoc(), getLhs(), getRhs(), getPrecisionConfig(),
                          getResult());
}

// PrecisionConfig - std::optional attribute, print the array as raw enums
//
// {precision_config = [#stablehlo<precision DEFAULT>,
//                      #stablehlo<precision DEFAULT>]}
// ==> ..., precision = [DEFAULT, DEFAULT]
void printPrecisionConfig(OpAsmPrinter& p, Operation*,
                          ::mlir::ArrayAttr attrArr) {
  // Precision config is an optional attribute, passes null if not specified.
  if (!attrArr) return;

  p << ", precision = [";
  llvm::interleaveComma(attrArr, p, [&](Attribute attr) {
    p << stringifyPrecision(attr.cast<PrecisionAttr>().getValue());
  });
  p << ']';
}

ParseResult parsePrecisionConfig(OpAsmParser& parser, mlir::ArrayAttr& attr) {
  // No precision config specified
  if (failed(parser.parseOptionalComma())) return success();

  if (failed(parser.parseKeyword("precision")) || failed(parser.parseEqual()))
    return failure();

  SmallVector<Attribute> attrs;
  if (failed(parser.parseCommaSeparatedList(
          AsmParser::Delimiter::Square, [&]() -> ParseResult {
            attrs.push_back(PrecisionAttr::parse(parser, {}));
            return success(/*isSuccess=*/static_cast<bool>(attrs.back()));
          })))
    return failure();

  attr = mlir::ArrayAttr::get(parser.getContext(), attrs);
  return success();
}

//===----------------------------------------------------------------------===//
// DotGeneralOp
//===----------------------------------------------------------------------===//

LogicalResult DotGeneralOp::verify() {
  return hlo::verifyDotGeneralOp(
      getLoc(), getLhs(), getRhs(),
      getDotDimensionNumbersAttr().getLhsBatchingDimensions(),
      getDotDimensionNumbersAttr().getRhsBatchingDimensions(),
      getDotDimensionNumbersAttr().getLhsContractingDimensions(),
      getDotDimensionNumbersAttr().getRhsContractingDimensions(),
      getPrecisionConfig(), getResult());
}

LogicalResult DotGeneralOp::reifyReturnTypeShapes(
    OpBuilder& builder, ValueRange operands,
    SmallVectorImpl<Value>& reifiedReturnShapes) {
  auto lhsType = getLhs().getType().dyn_cast<ShapedType>();
  auto rhsType = getRhs().getType().dyn_cast<ShapedType>();
  if (!lhsType || !rhsType) return failure();

  Adaptor adaptor(operands);
  auto dimNumbers = getDotDimensionNumbers();
  SmallVector<Value> dimensions;
  for (const int64_t lhsDim : dimNumbers.getLhsBatchingDimensions())
    dimensions.push_back(
        builder.create<tensor::DimOp>(getLoc(), adaptor.getLhs(), lhsDim));

  for (int64_t i = 0; i < lhsType.getRank(); i++)
    if (!llvm::is_contained(dimNumbers.getLhsContractingDimensions(), i) &&
        !llvm::is_contained(dimNumbers.getLhsBatchingDimensions(), i))
      dimensions.push_back(
          builder.create<tensor::DimOp>(getLoc(), adaptor.getLhs(), i));
  for (int64_t i = 0; i < rhsType.getRank(); i++)
    if (!llvm::is_contained(dimNumbers.getRhsContractingDimensions(), i) &&
        !llvm::is_contained(dimNumbers.getRhsBatchingDimensions(), i))
      dimensions.push_back(
          builder.create<tensor::DimOp>(getLoc(), adaptor.getRhs(), i));

  reifiedReturnShapes.push_back(
      builder.create<tensor::FromElementsOp>(getLoc(), dimensions));
  return success();
}

//===----------------------------------------------------------------------===//
// FftOp
//===----------------------------------------------------------------------===//

LogicalResult FftOp::inferReturnTypeComponents(
    MLIRContext*, std::optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  FftOp::Adaptor adaptor(operands, attributes, regions);
  return hlo::inferFftOp(location, adaptor.getOperand(),
                         adaptor.getFftType() == FftType::RFFT,
                         adaptor.getFftType() == FftType::IRFFT,
                         adaptor.getFftLength(), inferredReturnShapes);
}

//===----------------------------------------------------------------------===//
// GatherOp
//===----------------------------------------------------------------------===//

namespace {

// following https://www.tensorflow.org/xla/operation_semantics#gather
// The bounds for the output array along dimension i is computed as follows:
// (1) If i is present in batch_dims (i.e. is equal to batch_dims[k] for some k)
// then we pick
// the corresponding dimension bounds out of start_indices.shape, skipping
// index_vector_dim
// (i.e. pick start_indices.shape.dims[k] if k < index_vector_dim and
// start_indices.shape.dims[k+1] otherwise).
// (2) If i is present in offset_dims (i.e. equal to offset_dims[k] for some k)
// then we pick
// the corresponding bound out of slice_sizes after accounting for
// collapsed_slice_dims
// (i.e. we pick adjusted_slice_sizes[k] where adjusted_slice_sizes is
// slice_sizes with the bounds at indices collapsed_slice_dims removed).

void getSliceSizeValues(GatherOp* gather, OpBuilder& builder, Location loc,
                        ValueRange operands,
                        SmallVectorImpl<Value>& sliceSizes) {
  for (int64_t val : gather->getSliceSizes().getValues<int64_t>())
    sliceSizes.push_back(builder.create<arith::ConstantIndexOp>(loc, val));
}

void getSliceSizeValues(DynamicGatherOp* /*dGather*/, OpBuilder& builder,
                        Location loc, ValueRange operands,
                        SmallVectorImpl<Value>& sliceSizeValues) {
  DynamicGatherOp::Adaptor adaptor(operands);
  Value sliceSizes = adaptor.getSliceSizes();
  auto sliceSizesTy = sliceSizes.getType().cast<ShapedType>();
  for (int64_t i = 0; i < sliceSizesTy.getDimSize(0); ++i) {
    Value idx = builder.create<arith::ConstantIndexOp>(loc, i);
    sliceSizeValues.push_back(
        builder.create<tensor::ExtractOp>(loc, sliceSizes, idx));
  }
}

template <typename Op>
LogicalResult reifyGatherShape(Op* op, OpBuilder& builder, ValueRange operands,
                               SmallVectorImpl<Value>& reifiedReturnShapes) {
  // No support for unranked gather output shape a.t.m.
  auto resultTy =
      op->getResult().getType().template dyn_cast<RankedTensorType>();
  if (!resultTy) return failure();

  typename Op::Adaptor adaptor(operands);
  Value startIndices = adaptor.getStartIndices();

  Location loc = op->getLoc();
  int resultRank = resultTy.getRank();
  Type shapeElTy = builder.getIndexType();
  auto toShapeElType = [&](Value v) {
    return maybeCastTo(builder, loc, v, shapeElTy);
  };

  SmallVector<Value, 4> sliceSizes;
  getSliceSizeValues(op, builder, loc, operands, sliceSizes);
  llvm::transform(sliceSizes, sliceSizes.begin(),
                  [&](Value v) { return toShapeElType(v); });

  auto getStartIndicesDim = [&](int64_t index) {
    return toShapeElType(
        builder.create<tensor::DimOp>(loc, startIndices, index));
  };
  SmallVector<Value, 4> shapeValues;
  auto getSliceDim = [&sliceSizes](int64_t index) -> Value {
    return sliceSizes[index];
  };
  hlo::reifyGatherDimSizes(resultRank, getStartIndicesDim, getSliceDim,
                           op->getDimensionNumbers().getOffsetDims(),
                           op->getDimensionNumbers().getCollapsedSliceDims(),
                           op->getDimensionNumbers().getStartIndexMap(),
                           op->getDimensionNumbers().getIndexVectorDim(),
                           shapeValues);

  Value outputShape = builder.create<tensor::FromElementsOp>(
      loc, RankedTensorType::get({resultRank}, shapeElTy), shapeValues);
  reifiedReturnShapes.push_back(outputShape);

  return success();
}

}  // namespace

LogicalResult GatherOp::reifyReturnTypeShapes(
    OpBuilder& builder, ValueRange operands,
    SmallVectorImpl<Value>& reifiedReturnShapes) {
  return reifyGatherShape(this, builder, operands, reifiedReturnShapes);
}

LogicalResult GatherOp::inferReturnTypeComponents(
    MLIRContext* context, std::optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  GatherOp::Adaptor adaptor(operands, attributes, regions);
  return hlo::inferGatherOp(
      location, adaptor.getOperand(), adaptor.getStartIndices(),
      adaptor.getDimensionNumbers().getOffsetDims(),
      adaptor.getDimensionNumbers().getCollapsedSliceDims(),
      adaptor.getDimensionNumbers().getStartIndexMap(),
      adaptor.getDimensionNumbers().getIndexVectorDim(),
      adaptor.getSliceSizes(), inferredReturnShapes);
}

//===----------------------------------------------------------------------===//
// DynamicGatherOp
//===----------------------------------------------------------------------===//

LogicalResult DynamicGatherOp::reifyReturnTypeShapes(
    OpBuilder& builder, ValueRange operands,
    SmallVectorImpl<Value>& reifiedReturnShapes) {
  return reifyGatherShape(this, builder, operands, reifiedReturnShapes);
}

LogicalResult DynamicGatherOp::inferReturnTypeComponents(
    MLIRContext* context, std::optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  DynamicGatherOp::Adaptor adaptor(operands, attributes, regions);
  return hlo::inferDynamicGatherOp(
      location, adaptor.getOperand(), adaptor.getStartIndices(),
      adaptor.getSliceSizes(), adaptor.getDimensionNumbers().getOffsetDims(),
      adaptor.getDimensionNumbers().getCollapsedSliceDims(),
      adaptor.getDimensionNumbers().getStartIndexMap(),
      adaptor.getDimensionNumbers().getIndexVectorDim(), inferredReturnShapes);
}

//===----------------------------------------------------------------------===//
// GetDimensionSizeOp
//===----------------------------------------------------------------------===//

LogicalResult GetDimensionSizeOp::inferReturnTypeComponents(
    MLIRContext*, std::optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  GetDimensionSizeOp::Adaptor adaptor(operands, attributes, regions);
  return hlo::inferGetDimensionSizeOp(location, adaptor.getOperand().getType(),
                                      adaptor.getDimension(),
                                      inferredReturnShapes);
}

//===----------------------------------------------------------------------===//
// IotaOp
//===----------------------------------------------------------------------===//

LogicalResult IotaOp::verify() {
  return hlo::verifyIotaOp(getLoc(), getIotaDimension(), getResult());
}

//===----------------------------------------------------------------------===//
// DynamicIotaOp
//===----------------------------------------------------------------------===//

static Value castToIndexTensor(OpBuilder& builder, Location loc,
                               Value shapeOp) {
  ShapedType resultTy = shape::getExtentTensorType(
      builder.getContext(), shapeOp.getType().cast<ShapedType>().getDimSize(0));
  if (shapeOp.getType() == resultTy) return shapeOp;  // Nothing to do.
  return builder.create<arith::IndexCastOp>(loc, resultTy, shapeOp);
}

LogicalResult DynamicIotaOp::reifyReturnTypeShapes(
    OpBuilder& builder, ValueRange operands,
    SmallVectorImpl<Value>& reifiedReturnShapes) {
  DynamicIotaOp::Adaptor adaptor(operands);
  reifiedReturnShapes.push_back(
      castToIndexTensor(builder, getLoc(), adaptor.getOutputShape()));
  return success();
}

//===----------------------------------------------------------------------===//
// DynamicUpdateSliceOp
//===----------------------------------------------------------------------===//

LogicalResult DynamicUpdateSliceOp::inferReturnTypeComponents(
    MLIRContext*, std::optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  DynamicUpdateSliceOp::Adaptor adaptor(operands, attributes, regions);
  return hlo::inferDynamicUpdateSliceOp(
      location, adaptor.getOperand(), adaptor.getUpdate(),
      adaptor.getStartIndices(), inferredReturnShapes);
}

//===----------------------------------------------------------------------===//
// AbsOp
//===----------------------------------------------------------------------===//

LogicalResult AbsOp::inferReturnTypes(
    MLIRContext*, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  AbsOp::Adaptor adaptor(operands, attributes, regions);
  return hlo::inferAbsOp(location, adaptor.getOperand(), inferredReturnTypes);
}

//===----------------------------------------------------------------------===//
// CollectivePermuteOp
//===----------------------------------------------------------------------===//

LogicalResult CollectivePermuteOp::verify() {
  return hlo::verifyCollectivePermuteOp(getLoc(), getSourceTargetPairs());
}

//===----------------------------------------------------------------------===//
// ConvolutionOp
//===----------------------------------------------------------------------===//

LogicalResult ConvolutionOp::verify() {
  return hlo::verifyConvolutionOp(
      getLoc(), getLhs().getType(), getRhs().getType(), getWindowStrides(),
      getPadding(), getLhsDilation(), getRhsDilation(), getWindowReversal(),
      getDimensionNumbers().getInputBatchDimension(),
      getDimensionNumbers().getInputFeatureDimension(),
      getDimensionNumbers().getInputSpatialDimensions(),
      getDimensionNumbers().getKernelInputFeatureDimension(),
      getDimensionNumbers().getKernelOutputFeatureDimension(),
      getDimensionNumbers().getKernelSpatialDimensions(),
      getDimensionNumbers().getOutputBatchDimension(),
      getDimensionNumbers().getOutputFeatureDimension(),
      getDimensionNumbers().getOutputSpatialDimensions(),
      getFeatureGroupCount(), getBatchGroupCount(), getPrecisionConfig(),
      getResult().getType());
}

//===----------------------------------------------------------------------===//
// ConvertOp
//===----------------------------------------------------------------------===//

void ConvertOp::build(OpBuilder& builder, OperationState& result, Value operand,
                      Type resultElementTy) {
  Type resultTy;
  Type operandTy = operand.getType();
  if (auto rankedTy = operandTy.dyn_cast<RankedTensorType>())
    resultTy = RankedTensorType::get(rankedTy.getShape(), resultElementTy);
  else
    resultTy = UnrankedTensorType::get(resultElementTy);
  build(builder, result, resultTy, operand);
}

//===----------------------------------------------------------------------===//
// AllToAllOp
//===----------------------------------------------------------------------===//

LogicalResult AllToAllOp::inferReturnTypeComponents(
    MLIRContext*, std::optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  AllToAllOp::Adaptor adaptor(operands, attributes, regions);
  return hlo::inferAllToAllOp(
      location, adaptor.getOperand(), adaptor.getSplitDimension(),
      adaptor.getConcatDimension(), adaptor.getSplitCount(),
      adaptor.getReplicaGroups(), inferredReturnShapes);
}

void AllToAllOp::build(OpBuilder& odsBuilder, OperationState& odsState,
                       Type resultType, Value operand,
                       IntegerAttr splitDimension, IntegerAttr concatDimension,
                       IntegerAttr splitCount,
                       DenseIntElementsAttr replicaGroups) {
  AllToAllOp::build(odsBuilder, odsState, resultType, operand, splitDimension,
                    concatDimension, splitCount, replicaGroups,
                    /*channel_handle=*/nullptr);
}

//===----------------------------------------------------------------------===//
// AllGatherOp
//===----------------------------------------------------------------------===//

LogicalResult AllGatherOp::verify() {
  return hlo::verifyAllGatherOp(getLoc(), getOperand(), getAllGatherDim(),
                                getReplicaGroups(), getUseGlobalDeviceIds(),
                                getResult());
}

//===----------------------------------------------------------------------===//
// AllReduceOp
//===----------------------------------------------------------------------===//

LogicalResult AllReduceOp::verify() {
  return hlo::verifyAllReduceOp(getLoc(), getOperand(), getReplicaGroups(),
                                getUseGlobalDeviceIds(), getComputation());
}

//===----------------------------------------------------------------------===//
// BatchNormGradOp
//===----------------------------------------------------------------------===//

LogicalResult BatchNormGradOp::inferReturnTypeComponents(
    MLIRContext* context, std::optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  BatchNormGradOp::Adaptor adaptor(operands, attributes, regions);
  return hlo::inferBatchNormGradOp(
      location, adaptor.getOperand(), adaptor.getScale(), adaptor.getMean(),
      adaptor.getVariance(), adaptor.getGradOutput(), adaptor.getFeatureIndex(),
      inferredReturnShapes);
}

//===----------------------------------------------------------------------===//
// BatchNormTrainingOp
//===----------------------------------------------------------------------===//

LogicalResult BatchNormTrainingOp::inferReturnTypeComponents(
    MLIRContext* context, std::optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  BatchNormTrainingOp::Adaptor adaptor(operands, attributes, regions);
  return hlo::inferBatchNormTrainingOp(
      location, adaptor.getOperand(), adaptor.getScale(), adaptor.getOffset(),
      adaptor.getFeatureIndex(), inferredReturnShapes);
}

//===----------------------------------------------------------------------===//
// BatchNormInferenceOp
//===----------------------------------------------------------------------===//

LogicalResult BatchNormInferenceOp::inferReturnTypeComponents(
    MLIRContext* context, std::optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  BatchNormInferenceOp::Adaptor adaptor(operands, attributes, regions);
  return hlo::inferBatchNormInferenceOp(
      location, adaptor.getOperand(), adaptor.getScale(), adaptor.getOffset(),
      adaptor.getMean(), adaptor.getVariance(), adaptor.getFeatureIndex(),
      inferredReturnShapes);
}

//===----------------------------------------------------------------------===//
// BitcastConvertOp
//===----------------------------------------------------------------------===//

LogicalResult BitcastConvertOp::reifyReturnTypeShapes(
    OpBuilder& builder, ValueRange operands,
    SmallVectorImpl<Value>& reifiedReturnShapes) {
  auto operandType = operands[0].getType().dyn_cast<RankedTensorType>();
  auto resultType = getType().dyn_cast<RankedTensorType>();

  // Only ranked tensors are supported.
  if (!operandType || !resultType) return failure();

  // Shape-changing bitcast convert is not implemented.
  // TODO(kramerb): This could be done by adjusting the last dimension.
  DataLayout dataLayout = DataLayout::closest(*this);
  unsigned operandElementSize =
      dataLayout.getTypeSizeInBits(operandType.getElementType());
  unsigned resultElementSize =
      dataLayout.getTypeSizeInBits(resultType.getElementType());
  if (operandElementSize != resultElementSize) return failure();

  return hlo::deriveShapeFromOperand(&builder, getOperation(), operands.front(),
                                     &reifiedReturnShapes);
}

LogicalResult BitcastConvertOp::verify() {
  return hlo::verifyBitcastConvertOp(getLoc(), getOperand(), getResult());
}

//===----------------------------------------------------------------------===//
// BroadcastOp
//===----------------------------------------------------------------------===//

LogicalResult BroadcastOp::inferReturnTypeComponents(
    MLIRContext*, std::optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  BroadcastOp::Adaptor adaptor(operands, attributes, regions);
  return hlo::inferBroadcastOp(location, adaptor.getOperand(),
                               adaptor.getBroadcastSizes(),
                               inferredReturnShapes);
}

LogicalResult BroadcastOp::reifyReturnTypeShapes(
    OpBuilder& builder, ValueRange operands,
    SmallVectorImpl<Value>& reifiedReturnShapes) {
  BroadcastOp::Adaptor adaptor(operands);
  Value operand = adaptor.getOperand();

  auto operandType = operand.getType().dyn_cast<RankedTensorType>();
  // Unranked tensors are not supported.
  if (!operandType) return failure();

  Location loc = getLoc();
  SmallVector<Value, 4> shapeValues;

  // Collect the broadcast sizes.
  for (const auto& size : getBroadcastSizes())
    shapeValues.push_back(
        builder.create<arith::ConstantIndexOp>(loc, size.getZExtValue()));

  // Collect the operand sizes.
  for (auto index : llvm::seq<int64_t>(0, operandType.getRank()))
    shapeValues.push_back(
        builder.createOrFold<tensor::DimOp>(loc, operand, index));

  reifiedReturnShapes.push_back(builder.create<tensor::FromElementsOp>(
      loc,
      RankedTensorType::get({static_cast<int64_t>(shapeValues.size())},
                            builder.getIndexType()),
      shapeValues));

  return success();
}

//===----------------------------------------------------------------------===//
// BroadcastInDimOp
//===----------------------------------------------------------------------===//

LogicalResult BroadcastInDimOp::verify() {
  return hlo::verifyBroadcastInDimOp(getLoc(), getOperand(),
                                     getBroadcastDimensions(), getResult());
}

//===----------------------------------------------------------------------===//
// DynamicBroadcastInDimOp
//===----------------------------------------------------------------------===//

LogicalResult DynamicBroadcastInDimOp::verify() {
  return hlo::verifyDynamicBroadcastInDimOp(
      getLoc(), getOperand(), getOutputDimensions(), getBroadcastDimensions(),
      getKnownExpandingDimensions(), getKnownNonexpandingDimensions(),
      getResult());
}

LogicalResult DynamicBroadcastInDimOp::reifyReturnTypeShapes(
    OpBuilder& builder, ValueRange operands,
    SmallVectorImpl<Value>& reifiedReturnShapes) {
  DynamicBroadcastInDimOp::Adaptor adaptor(operands);
  reifiedReturnShapes.push_back(
      castToIndexTensor(builder, getLoc(), adaptor.getOutputDimensions()));
  return success();
}

//===----------------------------------------------------------------------===//
// ClampOp
//===----------------------------------------------------------------------===//

LogicalResult ClampOp::inferReturnTypeComponents(
    MLIRContext*, std::optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  ClampOp::Adaptor adaptor(operands, attributes, regions);
  return hlo::inferClampOp(location, adaptor.getMin(), adaptor.getOperand(),
                           adaptor.getMax(), inferredReturnShapes);
}

LogicalResult ClampOp::reifyReturnTypeShapes(
    OpBuilder& builder, ValueRange operands,
    SmallVectorImpl<Value>& reifiedReturnShapes) {
  // For `stablehlo.clamp`, the first operand may be a scalar.
  return hlo::deriveShapeFromOperand(&builder, getOperation(), operands[1],
                                     &reifiedReturnShapes);
}

//===----------------------------------------------------------------------===//
// ComplexOp
//===----------------------------------------------------------------------===//

LogicalResult ComplexOp::inferReturnTypes(
    MLIRContext*, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  ComplexOp::Adaptor adaptor(operands, attributes, regions);
  return hlo::inferComplexOp(location, adaptor.getLhs(), inferredReturnTypes);
}

//===----------------------------------------------------------------------===//
// ImagOp
//===----------------------------------------------------------------------===//

LogicalResult ImagOp::inferReturnTypes(
    MLIRContext*, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  ImagOp::Adaptor adaptor(operands, attributes, regions);
  return hlo::inferImagOp(location, adaptor.getOperand(), inferredReturnTypes);
}

//===----------------------------------------------------------------------===//
// IsFiniteOp
//===----------------------------------------------------------------------===//

LogicalResult IsFiniteOp::inferReturnTypes(
    MLIRContext* ctx, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  IsFiniteOp::Adaptor adaptor(operands, attributes, regions);
  return hlo::inferIsFiniteOp(ctx, location, adaptor.getX(),
                              inferredReturnTypes);
}

//===----------------------------------------------------------------------===//
// RealOp
//===----------------------------------------------------------------------===//

LogicalResult RealOp::inferReturnTypes(
    MLIRContext*, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  RealOp::Adaptor adaptor(operands, attributes, regions);
  return hlo::inferRealOp(location, adaptor.getOperand(), inferredReturnTypes);
}

//===----------------------------------------------------------------------===//
// ConcatenateOp
//===----------------------------------------------------------------------===//

LogicalResult ConcatenateOp::inferReturnTypes(
    MLIRContext*, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  ConcatenateOp::Adaptor adaptor(operands, attributes, regions);
  return hlo::inferConcatenateOp(location, adaptor.getInputs().getTypes(),
                                 adaptor.getDimension(), inferredReturnTypes);
}

LogicalResult ConcatenateOp::reifyReturnTypeShapes(
    OpBuilder& builder, ValueRange operands,
    SmallVectorImpl<Value>& reifiedReturnShapes) {
  ConcatenateOp::Adaptor adaptor(operands);
  auto inputs = adaptor.getInputs();

  auto operandType = inputs[0].getType().dyn_cast<RankedTensorType>();
  // Not support unranked type a.t.m.
  if (!operandType) return failure();

  Location loc = this->getLoc();
  Type shapeScalarType = builder.getIndexType();
  auto toShapeScalarType = [&](Value v) {
    return maybeCastTo(builder, loc, v, shapeScalarType);
  };

  SmallVector<SmallVector<Value, 4>, 4> allShapeValues;
  for (size_t inputId = 0; inputId < inputs.size(); ++inputId) {
    Value operand = inputs[inputId];
    auto operandType = operand.getType().dyn_cast<RankedTensorType>();
    if (!operandType) return failure();

    SmallVector<Value, 4> shapeVals;
    for (const auto& element : llvm::enumerate(operandType.getShape())) {
      Value valueDim = toShapeScalarType(
          builder.create<tensor::DimOp>(loc, operand, element.index()));
      shapeVals.push_back(valueDim);
    }
    allShapeValues.emplace_back(std::move(shapeVals));
  }

  int axis = this->getDimension();
  auto& shapeValues = allShapeValues[0];
  for (size_t vecId = 1; vecId < allShapeValues.size(); ++vecId) {
    auto& otherShapeValues = allShapeValues[vecId];
    if (otherShapeValues.size() != shapeValues.size()) {
      this->emitOpError()
          << "Concatenate expects all operands must be of the same rank";
      return failure();
    }
    shapeValues[axis] = builder.create<arith::AddIOp>(loc, shapeValues[axis],
                                                      otherShapeValues[axis]);
  }

  Value outputShape = builder.create<tensor::FromElementsOp>(
      loc,
      RankedTensorType::get({static_cast<int64_t>(shapeValues.size())},
                            shapeScalarType),
      shapeValues);
  reifiedReturnShapes.push_back(outputShape);

  return success();
}

//===----------------------------------------------------------------------===//
// DynamicReshapeOp
//===----------------------------------------------------------------------===//

LogicalResult DynamicReshapeOp::verify() {
  return hlo::verifyDynamicReshapeOp(getLoc(), getOutputShape(), getResult());
}

LogicalResult DynamicReshapeOp::reifyReturnTypeShapes(
    OpBuilder& builder, ValueRange operands,
    SmallVectorImpl<Value>& reifiedReturnShapes) {
  DynamicReshapeOp::Adaptor adaptor(operands);
  reifiedReturnShapes.push_back(
      castToIndexTensor(builder, getLoc(), adaptor.getOutputShape()));
  return success();
}

//===----------------------------------------------------------------------===//
// DynamicSliceOp
//===----------------------------------------------------------------------===//

LogicalResult DynamicSliceOp::inferReturnTypeComponents(
    MLIRContext*, std::optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  DynamicSliceOp::Adaptor adaptor(operands, attributes, regions);
  return hlo::inferDynamicSliceOp(location, adaptor.getOperand().getType(),
                                  adaptor.getStartIndices().getTypes(),
                                  adaptor.getSliceSizes(),
                                  inferredReturnShapes);
}

//===----------------------------------------------------------------------===//
// RealDynamicSliceOp
//===----------------------------------------------------------------------===//
LogicalResult RealDynamicSliceOp::verify() {
  return hlo::verifyRealDynamicSliceOp(getLoc(), getOperand(),
                                       getStartIndices(), getLimitIndices(),
                                       getStrides());
}

LogicalResult RealDynamicSliceOp::reifyReturnTypeShapes(
    OpBuilder& builder, ValueRange operands,
    SmallVectorImpl<Value>& reifiedReturnShapes) {
  RealDynamicSliceOp::Adaptor adaptor(operands);
  Value operand = adaptor.getOperand();
  Value startIndices = adaptor.getStartIndices();
  Value limitIndices = adaptor.getLimitIndices();
  Value strides = adaptor.getStrides();

  auto operandType = operand.getType().dyn_cast<RankedTensorType>();
  // Not support unranked type a.t.m.
  if (!operandType) return failure();

  Location loc = this->getLoc();
  SmallVector<Value, 4> shapeValues;
  shapeValues.reserve(operandType.getRank());
  Type shapeScalarType =
      startIndices.getType().cast<ShapedType>().getElementType();
  Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
  one = maybeCastTo(builder, loc, one, shapeScalarType);
  for (const auto& element : llvm::enumerate(operandType.getShape())) {
    Value offset = builder.create<arith::ConstantIndexOp>(loc, element.index());
    Value valueStart =
        builder.create<tensor::ExtractOp>(loc, startIndices, offset);
    Value valueLimit =
        builder.create<tensor::ExtractOp>(loc, limitIndices, offset);
    Value valueStride = builder.create<tensor::ExtractOp>(loc, strides, offset);
    // size = (limit - start + stride - 1) / stride
    shapeValues.push_back(builder.create<arith::DivSIOp>(
        loc,
        builder.create<arith::SubIOp>(
            loc,
            builder.create<arith::AddIOp>(
                loc, valueStride,
                builder.create<arith::SubIOp>(loc, valueLimit, valueStart)),
            one),
        valueStride));
  }

  reifiedReturnShapes.push_back(builder.create<tensor::FromElementsOp>(
      loc,
      RankedTensorType::get({static_cast<int64_t>(shapeValues.size())},
                            shapeScalarType),
      shapeValues));
  return success();
}

//===----------------------------------------------------------------------===//
// InfeedOp
//===----------------------------------------------------------------------===//

LogicalResult InfeedOp::verify() {
  return hlo::verifyInfeedOp(getStablehloDialect(getContext()), getLoc(),
                             getLayout(), getResults());
}

//===----------------------------------------------------------------------===//
// MapOp
//===----------------------------------------------------------------------===//

LogicalResult MapOp::inferReturnTypeComponents(
    MLIRContext*, std::optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  MapOp::Adaptor adaptor(operands, attributes, regions);
  return hlo::inferMapOp(location, adaptor.getInputs(), adaptor.getDimensions(),
                         adaptor.getComputation(), inferredReturnShapes);
}

LogicalResult MapOp::reifyReturnTypeShapes(
    OpBuilder& builder, ValueRange operands,
    SmallVectorImpl<Value>& reifiedReturnShapes) {
  return hlo::deriveShapeFromOperand(&builder, getOperation(), operands.front(),
                                     &reifiedReturnShapes);
}

//===----------------------------------------------------------------------===//
// OutfeedOp
//===----------------------------------------------------------------------===//

LogicalResult OutfeedOp::inferReturnTypes(
    MLIRContext* context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  return hlo::inferOutfeedOp(getStablehloDialect(context), location,
                             inferredReturnTypes);
}

//===----------------------------------------------------------------------===//
// Send Op
//===----------------------------------------------------------------------===//

LogicalResult SendOp::inferReturnTypes(
    MLIRContext* context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr, RegionRange, SmallVectorImpl<Type>& inferredReturnTypes) {
  return hlo::inferSendOp(getStablehloDialect(context), location,
                          inferredReturnTypes);
}

//===----------------------------------------------------------------------===//
// RecvOp
//===----------------------------------------------------------------------===//

LogicalResult RecvOp::verify() {
  return hlo::verifyRecvOp(getStablehloDialect(getContext()), getLoc(),
                           getResults());
}

//===----------------------------------------------------------------------===//
// ReduceWindowOp
//===----------------------------------------------------------------------===//

LogicalResult ReduceWindowOp::inferReturnTypeComponents(
    MLIRContext*, std::optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  ReduceWindowOp::Adaptor adaptor(operands, attributes, regions);
  return hlo::inferReduceWindowOp(
      location, adaptor.getInputs(), adaptor.getInitValues(),
      adaptor.getWindowDimensions(), adaptor.getWindowStrides(),
      adaptor.getBaseDilations(), adaptor.getWindowDilations(),
      adaptor.getPadding(), inferredReturnShapes);
}

LogicalResult ReduceWindowOp::verify() {
  return hlo::verifyReduceWindowOp(getLoc(), getInputs(), getInitValues(),
                                   getWindowDimensions(), getWindowStrides(),
                                   getBaseDilations(), getWindowDilations(),
                                   getPadding(), getBody());
}

// Builder that takes a constructor for its region and infers result types
void ReduceWindowOp::build(
    OpBuilder& odsBuilder, OperationState& odsState, ValueRange inputs,
    ValueRange init_values, DenseIntElementsAttr window_dimensions,
    /*optional*/ DenseIntElementsAttr window_strides,
    /*optional*/ DenseIntElementsAttr base_dilations,
    /*optional*/ DenseIntElementsAttr window_dilations,
    /*optional*/ DenseIntElementsAttr padding,
    function_ref<void(OpBuilder&, Location, ValueRange)> bodyBuilder) {
  odsState.addOperands(inputs);
  odsState.addOperands(init_values);
  odsState.addAttribute(getWindowDimensionsAttrName(odsState.name),
                        window_dimensions);
  if (window_strides)
    odsState.addAttribute(getWindowStridesAttrName(odsState.name),
                          window_strides);
  if (base_dilations)
    odsState.addAttribute(getBaseDilationsAttrName(odsState.name),
                          base_dilations);
  if (window_dilations)
    odsState.addAttribute(getWindowDilationsAttrName(odsState.name),
                          window_dilations);
  if (padding)
    odsState.addAttribute(getPaddingAttrName(odsState.name), padding);
  Region* region = odsState.addRegion();

  llvm::SmallVector<Type> blockArgTypes;
  llvm::SmallVector<Location> locs;
  auto numValues = inputs.size() + init_values.size();
  blockArgTypes.reserve(numValues);
  locs.reserve(numValues);
  for (auto i : inputs) {
    auto iType = i.getType().cast<ShapedType>();
    blockArgTypes.push_back(iType.cloneWith(
        llvm::ArrayRef<int64_t>(std::nullopt), iType.getElementType()));
    locs.push_back(i.getLoc());
  }
  for (auto i : init_values) {
    auto iType = i.getType().cast<ShapedType>();
    blockArgTypes.push_back(iType.cloneWith(
        llvm::ArrayRef<int64_t>(std::nullopt), iType.getElementType()));
    locs.push_back(i.getLoc());
  }

  {
    OpBuilder::InsertionGuard g(odsBuilder);
    Block* body =
        odsBuilder.createBlock(region, /*insertPt=*/{}, blockArgTypes, locs);
    bodyBuilder(odsBuilder, odsState.location, body->getArguments());
  }

  llvm::SmallVector<mlir::Type, 2> inferredReturnTypes;
  if (mlir::succeeded(ReduceWindowOp::inferReturnTypes(
          odsBuilder.getContext(), odsState.location, odsState.operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes)))
    odsState.addTypes(inferredReturnTypes);
  else
    llvm::report_fatal_error("Failed to infer result type(s).");
}

//===----------------------------------------------------------------------===//
// ReducePrecisionOp
//===----------------------------------------------------------------------===//

LogicalResult ReducePrecisionOp::verify() {
  return hlo::verifyReducePrecisionOp(getLoc(), getExponentBits(),
                                      getMantissaBits());
}

//===----------------------------------------------------------------------===//
// ReduceOp
//===----------------------------------------------------------------------===//

bool hasSameOperandAndResultTypes(Operation& op) {
  Type expected;
  if (op.getNumResults() != 0) expected = op.getResult(0).getType();
  if (op.getNumOperands() != 0) expected = op.getOperand(0).getType();
  if (!expected) return false;

  auto typeMatch = [&](Type actual) { return actual == expected; };
  return llvm::all_of(op.getOperandTypes(), typeMatch) &&
         llvm::all_of(op.getResultTypes(), typeMatch);
}

// Checks the following eligibility criteria for compact printing of reduce:
// E1. The reduce-op wraps a single inner-op in the associated region.
// E2. The single operation is a commutative binary-op from the dialect, zero
//     region, producing single result such that the operands and result all
//     have the same type.
// E3. The reduce-op consist of at least one input-operand; The operand-types of
//     inner-op should be derived trivially from the element-type of reduce-op's
//     first input-operand.
// E4. The  arguments of the region's only basic block are forwarded perfectly
//     to inner-op's operands.
// E5. The reduce-op, inner-op, blocks arguments, and the return-op all have the
//     same location.
// E6. The single operation result is perfectly forwarded to the reduce op
//     return.
static bool isEligibleForCompactPrint(ReduceOp op) {
  // Check E1.
  auto& block = op.getBody().front();
  if (!hasSingleElement(block.without_terminator())) return false;

  Operation& innerOp = *block.begin();

  // Check E2.
  if (innerOp.getDialect() != op->getDialect()) return false;

  if (innerOp.getNumOperands() != 2 ||
      !innerOp.hasTrait<mlir::OpTrait::OneResult>() ||
      !hasSameOperandAndResultTypes(innerOp) ||
      !innerOp.hasTrait<mlir::OpTrait::IsCommutative>() ||
      !innerOp.hasTrait<mlir::OpTrait::ZeroRegions>())
    return false;

  // Check E3.
  if (op.getInputs().empty()) return false;

  auto elemType =
      op.getInputs()[0].getType().cast<TensorType>().getElementType();
  auto expectedInnerOpType = RankedTensorType::get(/*shape=*/{}, elemType);
  if (innerOp.getOperands()[0].getType() != expectedInnerOpType) return false;

  // Check E4.
  if (!llvm::equal(block.getArguments(), innerOp.getOperands())) return false;

  // Check E5.
  auto retOp = dyn_cast<ReturnOp>(block.getTerminator());
  if (!retOp) return false;

  auto blockArgLoc = block.getArgument(0).getLoc();
  if (blockArgLoc != block.getArgument(1).getLoc()) return false;

  if (innerOp.getLoc() != op.getLoc() || retOp.getLoc() != op.getLoc() ||
      blockArgLoc != op.getLoc())
    return false;

  // Check E6.
  return llvm::equal(innerOp.getResults(), retOp.getOperands());
}

void ReduceOp::print(OpAsmPrinter& p) {
  {
    // Print the pairs of operands under the form:
    //   (%arg0 init: %arg3), (%arg1 init: %arg4), (%arg2 init: %arg5)
    StringRef comma = "";
    int numOperandPairs = getNumOperands() / 2;
    for (int opId : llvm::seq<int>(0, numOperandPairs)) {
      p << comma << "(" << getOperand(opId)
        << " init: " << getOperand(opId + numOperandPairs) << ")";
      comma = ", ";
    }
  }

  // If the reduce-op is eligible for compact printing, we emit the one-liner:
  // stablehlo.reduce applies <inner-op> across dimensions = [...] : <func-type>
  // Note: We are not printing the function type of reduction operation. We
  // have some simplifying assumptions (refer to IsEligibleForCompactPrint::E3)
  // to derive the type from that of reduce-op.
  if (isEligibleForCompactPrint(*this)) {
    Operation& innerOp = getBody().front().front();
    p << " applies ";
    printEscapedString(innerOp.getName().getStringRef(), p.getStream());

    p << " across dimensions = [";
    llvm::interleaveComma(getDimensions().getValues<int64_t>(), p);
    p << "]";
    p << " : ";
    p.printFunctionalType(*this);
  } else {
    p << " across dimensions = [";
    llvm::interleaveComma(getDimensions().getValues<int64_t>(), p);
    p << "]";
    p.printOptionalAttrDict(getOperation()->getAttrs(), {"dimensions"});
    p << " : ";
    p.printFunctionalType(*this);
    p.printNewline();
    p << " reducer";
    {
      // Print the pairs of block operands under the form:
      //   (%arg0_elt, %arg0_acc) (%arg1_elt, %arg1_acc):
      Block& reducer = getBody().front();
      int numOperandPairs = getNumOperands() / 2;
      for (int opId : llvm::seq<int>(0, numOperandPairs)) {
        p << "(";
        p.printRegionArgument(reducer.getArgument(opId));
        p << ", ";
        p.printRegionArgument(reducer.getArgument(opId + numOperandPairs));
        p << ") ";
      }
    }
    p << ' ';
    p.printRegion(getBody(), /*printEntryBlockArgs=*/false);
  }
}

ParseResult ReduceOp::parse(OpAsmParser& parser, OperationState& result) {
  llvm::SMLoc loc = parser.getCurrentLocation();
  Location currLocation = parser.getEncodedSourceLoc(loc);

  // Parse the operands of reduce-op, this is a list of pair under the form:
  //   (%arg0 init: %arg3), (%arg1 init: %arg4), (%arg2 init: %arg5)
  // Each input to reduce is paired with its init value, even though in memory
  // they are stored with the input first and the init values after.
  SmallVector<OpAsmParser::UnresolvedOperand, 2> operands;
  SmallVector<OpAsmParser::UnresolvedOperand, 2> initOperands;
  do {
    (void)parser.parseOptionalComma();
    if (parser.parseOptionalLParen()) break;
    OpAsmParser::UnresolvedOperand operand, initOperand;
    if (parser.parseOperand(operand) || parser.parseKeyword("init") ||
        parser.parseColon() || parser.parseOperand(initOperand) ||
        parser.parseRParen())
      return failure();
    operands.push_back(operand);
    initOperands.push_back(initOperand);
  } while (true);
  operands.append(initOperands);

  // Check if we are parsing the compact version of reduce-op:
  // stablehlo.reduce applies <inner-op> across dimensions = [...] : <func-type>
  // else parse the "region-based" variant.
  if (failed(parser.parseOptionalKeyword("applies"))) {
    // Parse the inner-op dimensions, reduce-op's function-type and
    // optional location.
    SmallVector<int64_t> dimensions;
    auto parseDim = [&]() -> ParseResult {
      if (parser.parseInteger(dimensions.emplace_back())) return failure();
      return success();
    };

    FunctionType reduceOpFntype;
    if (parser.parseKeyword("across") || parser.parseKeyword("dimensions") ||
        parser.parseEqual() ||
        parser.parseCommaSeparatedList(AsmParser::Delimiter::Square,
                                       parseDim) ||
        parser.parseOptionalAttrDict(result.attributes) ||
        parser.parseColon() || parser.parseType(reduceOpFntype) ||
        parser.parseKeyword("reducer"))
      return failure();
    OpBuilder builder(parser.getBuilder().getContext());
    result.addAttribute("dimensions", builder.getI64TensorAttr(dimensions));

    // Parse the "reducer" region now.
    SmallVector<OpAsmParser::UnresolvedOperand, 2> reducerOperands;
    SmallVector<OpAsmParser::UnresolvedOperand, 2> reducerInitOperands;
    SmallVector<Type, 2> reducerTypes;
    SmallVector<Type, 2> reducerInitTypes;
    SmallVector<std::optional<Location>, 2> reducerLocs;
    SmallVector<std::optional<Location>, 2> reducerInitLocs;
    auto parseBlockOperand =
        [&](SmallVectorImpl<OpAsmParser::UnresolvedOperand>& operands,
            SmallVectorImpl<Type>& types,
            SmallVectorImpl<std::optional<Location>>& locs) -> ParseResult {
      OpAsmParser::UnresolvedOperand operand;
      Type type;
      std::optional<Location> loc;
      if (parser.parseOperand(operand, /*allowResultNumber=*/false) ||
          parser.parseColon() || parser.parseType(type) ||
          parser.parseOptionalLocationSpecifier(loc))
        return failure();
      operands.push_back(operand);
      types.push_back(type);
      locs.push_back(loc);
      return success();
    };
    do {
      if (failed(parser.parseOptionalLParen())) break;
      if (parseBlockOperand(reducerOperands, reducerTypes, reducerLocs) ||
          parser.parseComma() ||
          parseBlockOperand(reducerInitOperands, reducerInitTypes,
                            reducerInitLocs) ||
          parser.parseRParen())
        return failure();
    } while (true);
    reducerOperands.append(reducerInitOperands);
    reducerTypes.append(reducerInitTypes);
    reducerLocs.append(reducerInitLocs);
    result.addTypes(reduceOpFntype.getResults());
    SmallVector<OpAsmParser::Argument> reducerArgs;
    createArgs(reducerOperands, reducerTypes, reducerArgs);

    // Derive the SSA-values for reduce-op's operands and parse the region, and
    // the optional trailing location.
    std::optional<Location> trailingLoc;
    if (parser.resolveOperands(operands, reduceOpFntype.getInputs(), loc,
                               result.operands) ||
        parser.parseRegion(*result.addRegion(), reducerArgs))
      return failure();
    // Set the individual block arguments.
    for (auto argAndLoc :
         llvm::zip(result.regions.front()->front().getArguments(), reducerLocs))
      if (std::get<1>(argAndLoc))
        std::get<0>(argAndLoc).setLoc(std::get<1>(argAndLoc).value());
    result.location = trailingLoc.value_or(currLocation);
    return success();
  }

  // Parse the inner-op name and check if the contract on inner-op
  // mentioned in "isEligibleForCompactPrint::E2" for pretty-priting is met.
  FailureOr<OperationName> innerOpNameInfo = parser.parseCustomOperationName();
  if (failed(innerOpNameInfo)) return failure();

  StringRef innerOpName = innerOpNameInfo->getStringRef();
  Dialect* innerOpDialect = innerOpNameInfo->getDialect();
  if (!innerOpDialect || !innerOpDialect->getNamespace().equals("stablehlo") ||
      !innerOpNameInfo->hasTrait<mlir::OpTrait::NOperands<2>::Impl>() ||
      !innerOpNameInfo->hasTrait<mlir::OpTrait::OneResult>() ||
      !innerOpNameInfo->hasTrait<mlir::OpTrait::IsCommutative>() ||
      !innerOpNameInfo->hasTrait<mlir::OpTrait::ZeroRegions>()) {
    parser.emitError(loc,
                     "expected the inner-op to be a commutative binary-op from "
                     "stablehlo dialect, zero region, producing single result");
    return failure();
  }

  // Parse the inner-op dimensions, reduce-op's function-type and
  // optional location.
  SmallVector<int64_t> dimensions;
  auto parseDim = [&]() -> ParseResult {
    if (parser.parseInteger(dimensions.emplace_back())) return failure();
    return success();
  };

  std::optional<Location> explicitLoc;
  FunctionType reduceOpFntype;
  if (parser.parseKeyword("across") || parser.parseKeyword("dimensions") ||
      parser.parseEqual() ||
      parser.parseCommaSeparatedList(AsmParser::Delimiter::Square, parseDim) ||
      parser.parseColon() || parser.parseType(reduceOpFntype) ||
      parser.parseOptionalLocationSpecifier(explicitLoc))
    return failure();

  if (!reduceOpFntype || reduceOpFntype.getInputs().empty()) {
    if (!reduceOpFntype) return parser.emitError(loc, "expected function type");
    return parser.emitError(loc,
                            "input types missing in reduce-op function type");
  }

  // If location of reduce-op is explicitly provided, then use it; Else use
  // the parser's current location.
  Location reduceOpLoc = explicitLoc.value_or(currLocation);

  // Derive the SSA-values for reduce-op's operands.
  if (parser.resolveOperands(operands, reduceOpFntype.getInputs(), loc,
                             result.operands))
    return failure();

  // Derive the type of inner-op from that of reduce-op's input operand.
  auto innerOpType = RankedTensorType::get(
      /*shape=*/{}, getElementTypeOrSelf(reduceOpFntype.getInput(0)));

  // Add a region for reduce-op.
  Region& region = *result.addRegion();

  // Create a basic-block inside reduce-op's region.
  Block& block = region.emplaceBlock();
  auto lhs = block.addArgument(innerOpType, reduceOpLoc);
  auto rhs = block.addArgument(innerOpType, reduceOpLoc);

  // Create and insert an "inner-op" operation in the block.
  OpBuilder builder(parser.getBuilder().getContext());
  builder.setInsertionPointToStart(&block);

  OperationState innerOpState(reduceOpLoc, innerOpName);
  innerOpState.operands.push_back(lhs);
  innerOpState.operands.push_back(rhs);
  innerOpState.addTypes(innerOpType);

  Operation* innerOp = builder.create(innerOpState);

  // Insert a return statement in the block returning the inner-op's result.
  builder.create<ReturnOp>(innerOp->getLoc(), innerOp->getResults());

  // Populate the reduce-op operation-state with result-type, location, and
  // dimension attribute.
  result.addTypes(reduceOpFntype.getResults());
  result.location = innerOp->getLoc();
  result.addAttribute("dimensions", builder.getI64TensorAttr(dimensions));

  return success();
}

LogicalResult ReduceOp::inferReturnTypeComponents(
    MLIRContext*, std::optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  ReduceOp::Adaptor adaptor(operands, attributes, regions);
  return hlo::inferReduceOp(location, adaptor.getInputs().getTypes(),
                            adaptor.getInitValues().getTypes(),
                            adaptor.getDimensions(), inferredReturnShapes);
}

LogicalResult ReduceOp::verify() {
  return hlo::verifyReduceOp(getLoc(), getInputs(), getInitValues(),
                             getDimensions(), getBody());
}

LogicalResult ReduceOp::reifyReturnTypeShapes(
    OpBuilder& builder, ValueRange operands,
    SmallVectorImpl<Value>& reifiedReturnShapes) {
  ReduceOp::Adaptor adaptor(operands);
  auto inputs = adaptor.getInputs();

  auto operandType = inputs[0].getType().dyn_cast<RankedTensorType>();
  // Not support unranked type a.t.m.
  if (!operandType) return failure();

  Location loc = this->getLoc();
  SmallVector<Value, 4> shapeValues;
  SmallVector<int64_t, 4> dimensions(
      this->getDimensions().getValues<int64_t>());
  shapeValues.reserve(operandType.getRank());
  Type shapeScalarType = builder.getIndexType();
  auto toShapeScalarType = [&](Value v) {
    return maybeCastTo(builder, loc, v, shapeScalarType);
  };

  for (const auto& element : llvm::enumerate(operandType.getShape())) {
    int64_t idx = element.index();
    auto* it = std::find(dimensions.begin(), dimensions.end(), idx);
    if (it != dimensions.end()) continue;
    Value valueDim = toShapeScalarType(
        builder.create<tensor::DimOp>(loc, inputs[0], element.index()));
    shapeValues.push_back(valueDim);
  }

  Value outputShape = builder.create<tensor::FromElementsOp>(
      loc,
      RankedTensorType::get({static_cast<int64_t>(shapeValues.size())},
                            shapeScalarType),
      shapeValues);
  for (size_t i = 0; i < inputs.size(); ++i)
    reifiedReturnShapes.push_back(outputShape);

  return success();
}

//===----------------------------------------------------------------------===//
// OptimizationBarrierOp
//===----------------------------------------------------------------------===//
LogicalResult OptimizationBarrierOp::inferReturnTypes(
    MLIRContext*, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  OptimizationBarrierOp::Adaptor adaptor(operands, attributes);
  return hlo::inferOptimizationBarrierOp(location, adaptor.getOperand(),
                                         inferredReturnTypes);
}

//===----------------------------------------------------------------------===//
// ReverseOp
//===----------------------------------------------------------------------===//
LogicalResult ReverseOp::verify() {
  return hlo::verifyReverseOp(getLoc(), getOperand(), getDimensions());
}

LogicalResult ReverseOp::inferReturnTypeComponents(
    MLIRContext* context, std::optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  ReverseOp::Adaptor adaptor(operands, attributes, regions);
  return hlo::inferReverseOp(location, adaptor.getOperand().getType(),
                             inferredReturnShapes);
}

//===----------------------------------------------------------------------===//
// RngBitGeneratorOp
//===----------------------------------------------------------------------===//

// Verify that input state has the same shape as output shape
LogicalResult RngBitGeneratorOp::verify() {
  return hlo::verifyRngBitGeneratorOp(getLoc(), getInitialState(),
                                      getOutputState());
}

//===----------------------------------------------------------------------===//
// RngOp
//===----------------------------------------------------------------------===//

LogicalResult RngOp::inferReturnTypeComponents(
    MLIRContext* context, std::optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  RngOp::Adaptor adaptor(operands, attributes, regions);
  return hlo::inferRngOp(
      location, adaptor.getA(), adaptor.getB(), adaptor.getShape(),
      adaptor.getRngDistribution() == RngDistribution::UNIFORM,
      inferredReturnShapes);
}

LogicalResult RngOp::reifyReturnTypeShapes(
    OpBuilder& builder, ValueRange operands,
    SmallVectorImpl<Value>& reifiedReturnShapes) {
  RngOp::Adaptor adaptor(operands);
  reifiedReturnShapes.push_back(
      castToIndexTensor(builder, getLoc(), adaptor.getShape()));
  return success();
}

//===----------------------------------------------------------------------===//
// SelectOp
//===----------------------------------------------------------------------===//

LogicalResult SelectOp::inferReturnTypeComponents(
    MLIRContext*, std::optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, RegionRange,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  SelectOp::Adaptor op(operands, attributes);
  return hlo::inferSelectOp(location, op.getPred(), op.getOnTrue(),
                            op.getOnFalse(), inferredReturnShapes);
}

LogicalResult SelectOp::reifyReturnTypeShapes(
    OpBuilder& builder, ValueRange operands,
    SmallVectorImpl<Value>& reifiedReturnShapes) {
  // For `hlo.select`, the first operand may be a scalar.
  return hlo::deriveShapeFromOperand(&builder, getOperation(), operands[1],
                                     &reifiedReturnShapes);
}

//===----------------------------------------------------------------------===//
// SetDimensionSizeOp
//===----------------------------------------------------------------------===//

LogicalResult SetDimensionSizeOp::inferReturnTypeComponents(
    MLIRContext* context, std::optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  SetDimensionSizeOp::Adaptor adaptor(operands, attributes, regions);
  return hlo::inferSetDimensionSizeOp(
      getStablehloDialect(context), location, adaptor.getOperand().getType(),
      adaptor.getSize(), adaptor.getDimension(), inferredReturnShapes);
}

//===----------------------------------------------------------------------===//
// PadOp
//===----------------------------------------------------------------------===//

LogicalResult PadOp::inferReturnTypes(
    MLIRContext*, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  PadOp::Adaptor adaptor(operands, attributes, regions);
  return hlo::inferPadOp(location, adaptor.getOperand().getType(),
                         adaptor.getPaddingValue().getType(),
                         adaptor.getEdgePaddingLow(),
                         adaptor.getEdgePaddingHigh(),
                         adaptor.getInteriorPadding(), inferredReturnTypes);
}

LogicalResult PadOp::reifyReturnTypeShapes(
    OpBuilder& builder, ValueRange operands,
    SmallVectorImpl<Value>& reifiedReturnShapes) {
  PadOp::Adaptor adaptor(operands, this->getOperation()->getAttrDictionary());
  auto loc = this->getLoc();
  Value operand = adaptor.getOperand();
  auto operandTy = operand.getType().cast<RankedTensorType>();

  llvm::SmallVector<int32_t> padHigh;
  llvm::SmallVector<int32_t> padLow;
  llvm::SmallVector<int32_t> padInterior;

  auto padHighAttr = adaptor.getEdgePaddingHigh();
  auto padLowAttr = adaptor.getEdgePaddingLow();
  auto padInteriorAttr = adaptor.getInteriorPadding();

  padHigh.reserve(padHighAttr.getNumElements());
  padLow.reserve(padLowAttr.getNumElements());
  padInterior.reserve(padInteriorAttr.getNumElements());

  for (const APInt& val : padHighAttr.getValues<APInt>())
    padHigh.push_back(val.getSExtValue());

  for (const APInt& val : padLowAttr.getValues<APInt>())
    padLow.push_back(val.getSExtValue());

  for (const APInt& val : padInteriorAttr.getValues<APInt>())
    padInterior.push_back(val.getSExtValue());

  Value one = builder.create<arith::ConstantIndexOp>(loc, 1).getResult();
  Value zero = builder.create<arith::ConstantIndexOp>(loc, 0).getResult();

  llvm::SmallVector<Value> dimensions;
  dimensions.reserve(operandTy.getRank());
  for (int i = 0, s = operandTy.getRank(); i < s; ++i) {
    Value padEdge =
        builder.create<arith::ConstantIndexOp>(loc, padHigh[i] + padLow[i]);

    // First we grab the initial interior size.
    Value dim = builder.create<tensor::DimOp>(loc, operand, i).getResult();

    // Compute the interior of the tensor and determine padding size.
    if (padInterior[i] > 0) {
      Value padInter =
          builder.create<arith::ConstantIndexOp>(loc, padInterior[i])
              .getResult();
      Value interior = builder.create<arith::SubIOp>(loc, dim, one).getResult();
      interior = builder.create<arith::MaxSIOp>(loc, interior, zero);
      interior = builder.create<arith::MulIOp>(loc, interior, padInter);
      dim = builder.create<arith::AddIOp>(loc, dim, interior).getResult();
    }

    // Then we add the padding on the edge of the tensor.
    dim = builder.create<arith::AddIOp>(loc, dim, padEdge).getResult();
    dimensions.push_back(dim);
  }

  Value dimensionTensor =
      builder.create<tensor::FromElementsOp>(loc, dimensions).getResult();
  reifiedReturnShapes.push_back(dimensionTensor);
  return success();
}

//===----------------------------------------------------------------------===//
// DynamicPadOp
//===----------------------------------------------------------------------===//

LogicalResult DynamicPadOp::verify() {
  return hlo::verifyDynamicPadOp(getLoc(), getOperand(), getPaddingValue(),
                                 getEdgePaddingLow(), getEdgePaddingHigh(),
                                 getInteriorPadding(), getResult());
}

LogicalResult DynamicPadOp::reifyReturnTypeShapes(
    OpBuilder& builder, ValueRange operands,
    SmallVectorImpl<Value>& reifiedReturnShapes) {
  DynamicPadOp::Adaptor adaptor(operands);
  Value operand = adaptor.getOperand();
  Value edgePaddingLow = adaptor.getEdgePaddingLow();
  Value edgePaddingHigh = adaptor.getEdgePaddingHigh();
  Value interiorPadding = adaptor.getInteriorPadding();

  auto operandType = operand.getType().dyn_cast<RankedTensorType>();
  // Not support unranked pad a.t.m.
  if (!operandType) return failure();

  auto loc = this->getLoc();
  SmallVector<Value, 4> shapeValues;
  shapeValues.reserve(operandType.getRank());
  Type shapeScalarType =
      edgePaddingLow.getType().cast<ShapedType>().getElementType();

  auto toShapeScalarType = [&](Value v) {
    return maybeCastTo(builder, loc, v, shapeScalarType);
  };

  Value zero =
      toShapeScalarType(builder.create<arith::ConstantIndexOp>(loc, 0));
  Value one = toShapeScalarType(builder.create<arith::ConstantIndexOp>(loc, 1));

  for (int idx : llvm::seq<int>(0, operandType.getShape().size())) {
    Value valueDim =
        toShapeScalarType(builder.create<tensor::DimOp>(loc, operand, idx));
    Value offset = builder.create<arith::ConstantIndexOp>(loc, idx);
    Value valueLow =
        builder.create<tensor::ExtractOp>(loc, edgePaddingLow, offset);
    Value valueHigh =
        builder.create<tensor::ExtractOp>(loc, edgePaddingHigh, offset);
    Value valueInterior =
        builder.create<tensor::ExtractOp>(loc, interiorPadding, offset);
    // output_size = input_size + padding_low + padding_high + interior *
    // max(input_size - 1, 0)
    Value valueDimLessThanOne = builder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::slt, valueDim, one);
    Value interiorSize = builder.create<arith::MulIOp>(
        loc, valueInterior,
        builder.create<mlir::arith::SelectOp>(
            loc, valueDimLessThanOne, zero,
            builder.create<arith::SubIOp>(loc, valueDim, one)));
    shapeValues.push_back(builder.create<arith::AddIOp>(
        loc,
        builder.create<arith::AddIOp>(
            loc, builder.create<arith::AddIOp>(loc, interiorSize, valueDim),
            valueLow),
        valueHigh));
  }

  reifiedReturnShapes.push_back(builder.create<tensor::FromElementsOp>(
      loc,
      RankedTensorType::get({static_cast<int64_t>(shapeValues.size())},
                            shapeScalarType),
      shapeValues));

  return success();
}

//===----------------------------------------------------------------------===//
// ReshapeOp
//===----------------------------------------------------------------------===//

LogicalResult ReshapeOp::verify() {
  return hlo::verifyReshapeOp(getLoc(), getOperand(), getResult());
}

//===----------------------------------------------------------------------===//
// ReplicaId Op
//===----------------------------------------------------------------------===//

LogicalResult ReplicaIdOp::inferReturnTypes(
    MLIRContext* context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr, RegionRange, SmallVectorImpl<Type>& inferredReturnTypes) {
  return hlo::inferReplicaIdOp(context, location, inferredReturnTypes);
}

//===----------------------------------------------------------------------===//
// PartitionId Op
//===----------------------------------------------------------------------===//

LogicalResult PartitionIdOp::inferReturnTypes(
    MLIRContext* context, std::optional<Location> location,
    ValueRange /*operands*/, DictionaryAttr, RegionRange,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  return hlo::inferPartitionIdOp(context, location, inferredReturnTypes);
}

//===----------------------------------------------------------------------===//
// If Op
//===----------------------------------------------------------------------===//

LogicalResult IfOp::inferReturnTypes(
    MLIRContext*, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  IfOp::Adaptor adaptor(operands, attributes, regions);
  return hlo::inferIfOp(location, adaptor.getPred(), adaptor.getRegions(),
                        inferredReturnTypes);
}

//===----------------------------------------------------------------------===//
// Case Op
//===----------------------------------------------------------------------===//

LogicalResult CaseOp::inferReturnTypes(
    MLIRContext*, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  CaseOp::Adaptor adaptor(operands, attributes, regions);
  return hlo::inferCaseOp(location, adaptor.getIndex(), adaptor.getRegions(),
                          inferredReturnTypes);
}

//===----------------------------------------------------------------------===//
// SliceOp
//===----------------------------------------------------------------------===//

LogicalResult SliceOp::inferReturnTypes(
    MLIRContext* context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  SliceOpAdaptor adaptor(operands, attributes);
  return hlo::inferSliceOp(location, adaptor.getOperand().getType(),
                           adaptor.getStartIndices(), adaptor.getLimitIndices(),
                           adaptor.getStrides(), inferredReturnTypes);
}

//===----------------------------------------------------------------------===//
// SortOp
//===----------------------------------------------------------------------===//

void SortOp::build(OpBuilder& builder, OperationState& state,
                   ValueRange operands, int64_t dimension, bool isStable) {
  state.addOperands(operands);
  state.addAttribute("dimension", builder.getI64IntegerAttr(dimension));
  state.addAttribute("is_stable", builder.getBoolAttr(isStable));

  for (Value operand : operands) state.addTypes(operand.getType());

  state.addRegion();
}

LogicalResult SortOp::inferReturnTypeComponents(
    MLIRContext*, std::optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  SortOp::Adaptor adaptor(operands, attributes, regions);
  return hlo::inferSortOp(location, adaptor.getInputs(), inferredReturnShapes);
}

LogicalResult SortOp::verify() {
  return hlo::verifySortOp(getLoc(), getInputs(), getDimension(),
                           getComparator());
}

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//

LogicalResult TransposeOp::reifyReturnTypeShapes(
    OpBuilder& builder, ValueRange operands,
    SmallVectorImpl<Value>& reifiedReturnShapes) {
  TransposeOp::Adaptor adaptor(operands);
  Value operand = adaptor.getOperand();

  auto operandType = operand.getType().dyn_cast<RankedTensorType>();
  // Not support unranked type a.t.m.
  if (!operandType) return failure();

  Location loc = this->getLoc();
  SmallVector<int64_t, 4> permutation(
      this->getPermutation().getValues<int64_t>());
  SmallVector<Value, 4> shapeValues(permutation.size());

  Type shapeScalarType = builder.getIndexType();
  auto toShapeScalarType = [&](Value v) {
    return maybeCastTo(builder, loc, v, shapeScalarType);
  };

  for (const auto& element : llvm::enumerate(operandType.getShape())) {
    int64_t idx = element.index();
    auto* it = std::find(permutation.begin(), permutation.end(), idx);
    Value valueDim = toShapeScalarType(
        builder.createOrFold<tensor::DimOp>(loc, operand, element.index()));
    shapeValues[std::distance(permutation.begin(), it)] = valueDim;
  }

  Value outputShape = builder.create<tensor::FromElementsOp>(
      loc,
      RankedTensorType::get({static_cast<int64_t>(shapeValues.size())},
                            shapeScalarType),
      shapeValues);
  reifiedReturnShapes.push_back(outputShape);

  return success();
}

LogicalResult TransposeOp::inferReturnTypes(
    MLIRContext*, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  TransposeOp::Adaptor adaptor(operands, attributes, regions);
  return hlo::inferTransposeOp(loc, adaptor.getOperand(),
                               adaptor.getPermutation(), inferredReturnTypes);
}

//===----------------------------------------------------------------------===//
// TriangularSolveOp
//===----------------------------------------------------------------------===//

LogicalResult TriangularSolveOp::inferReturnTypeComponents(
    MLIRContext*, std::optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  TriangularSolveOp::Adaptor adaptor(operands, attributes, regions);
  bool isTransposeAInvalid =
      (adaptor.getTransposeA() == Transpose::TRANSPOSE_INVALID);
  return hlo::inferTriangularSolveOp(location, adaptor.getA(), adaptor.getB(),
                                     adaptor.getLeftSide(), isTransposeAInvalid,
                                     inferredReturnShapes);
}

//===----------------------------------------------------------------------===//
// GetTupleElementOp
//===----------------------------------------------------------------------===//

LogicalResult GetTupleElementOp::inferReturnTypes(
    MLIRContext*, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  GetTupleElementOp::Adaptor adaptor(operands, attributes, regions);
  return hlo::inferGetTupleElementOp(location, adaptor.getOperand(),
                                     adaptor.getIndex(), inferredReturnTypes);
}

//===----------------------------------------------------------------------===//
// TupleOp
//===----------------------------------------------------------------------===//

LogicalResult TupleOp::inferReturnTypes(
    MLIRContext* context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  TupleOp::Adaptor adaptor(operands, attributes, regions);
  return hlo::inferTupleOp(context, location, adaptor.getVal(),
                           inferredReturnTypes);
}

//===----------------------------------------------------------------------===//
// CompareOp
//===----------------------------------------------------------------------===//

void CompareOp::build(OpBuilder& builder, OperationState& result, Value lhs,
                      Value rhs, ComparisonDirection comparisonDirection,
                      ComparisonType compareType) {
  build(builder, result, lhs, rhs,
        ComparisonDirectionAttr::get(builder.getContext(), comparisonDirection),
        ComparisonTypeAttr::get(builder.getContext(), compareType));
}

LogicalResult CompareOp::inferReturnTypeComponents(
    MLIRContext* context, std::optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  CompareOp::Adaptor adaptor(operands, attributes, regions);
  return hlo::inferCompareOp(context, location, adaptor.getLhs(),
                             inferredReturnShapes);
}

LogicalResult CompareOp::reifyReturnTypeShapes(
    OpBuilder& builder, ValueRange operands,
    SmallVectorImpl<Value>& reifiedReturnShapes) {
  return hlo::deriveShapeFromOperand(&builder, getOperation(), operands.front(),
                                     &reifiedReturnShapes);
}

//===----------------------------------------------------------------------===//
// SelectAndScatterOp
//===----------------------------------------------------------------------===//

LogicalResult SelectAndScatterOp::inferReturnTypes(
    MLIRContext*, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  SelectAndScatterOp::Adaptor adaptor(operands, attributes, regions);
  return hlo::inferSelectAndScatterOp(adaptor.getOperand(),
                                      inferredReturnTypes);
}

LogicalResult SelectAndScatterOp::verify() {
  return hlo::verifySelectAndScatterOp(getLoc(), getOperand(), getSource(),
                                       getInitValue(), getWindowDimensions(),
                                       getWindowStrides(), getPadding(),
                                       getSelect(), getScatter());
}

//===----------------------------------------------------------------------===//
// ScatterOp
//===----------------------------------------------------------------------===//

LogicalResult ScatterOp::inferReturnTypes(
    MLIRContext*, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  ScatterOp::Adaptor adaptor(operands, attributes, regions);
  return hlo::inferScatterOp(location, adaptor.getInputs(),
                             inferredReturnTypes);
}

LogicalResult ScatterOp::verify() {
  return hlo::verifyScatterOp(
      getLoc(), getInputs(), getScatterIndices(), getUpdates(),
      getScatterDimensionNumbers().getUpdateWindowDims(),
      getScatterDimensionNumbers().getInsertedWindowDims(),
      getScatterDimensionNumbers().getScatterDimsToOperandDims(),
      getScatterDimensionNumbers().getIndexVectorDim(), getUpdateComputation());
}

//===----------------------------------------------------------------------===//
// WhileOp
//===----------------------------------------------------------------------===//

LogicalResult WhileOp::inferReturnTypes(
    MLIRContext*, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  WhileOp::Adaptor adaptor(operands, attributes, regions);
  return hlo::inferWhileOp(location, adaptor.getOperand(), inferredReturnTypes);
}

LogicalResult WhileOp::verify() {
  return hlo::verifyWhileOp(getLoc(), getOperand(), getCond(), getBody());
}

/// Print a `while` op.
///
/// op ::= `stablehlo.while` `(` assignment-list `)` `:` types attribute-dict
///         `cond` region
///         `do` region
/// assignment-list ::= assignment | assignment `,` assignment-list
/// assignment ::= ssa-value `=` ssa-value
void WhileOp::print(OpAsmPrinter& p) {
  p << '(';
  llvm::interleaveComma(
      llvm::zip(SingleBlock::getBody()->getArguments(), getOperands()), p,
      [&](auto zip) {
        p.printOperand(std::get<0>(zip));
        p << " = ";
        p.printOperand(std::get<1>(zip));
      });
  p << ")";
  if (getNumOperands()) {
    p << " : ";
    llvm::interleaveComma(getOperandTypes(), p);
  }
  p.printOptionalAttrDictWithKeyword(getOperation()->getAttrs());
  p.printNewline();
  p << " cond ";
  p.printRegion(getRegion(0), /*printEntryBlockArgs=*/false);
  p << " do ";
  p.printRegion(getRegion(1), /*printEntryBlockArgs=*/false);
}

ParseResult WhileOp::parse(OpAsmParser& parser, OperationState& result) {
  llvm::SMLoc loc = parser.getCurrentLocation();
  // Parse the operands of the while: these are of the form:
  //   %iter_arg = %init_val
  // where %iter_arg is the name of the block argument in the cond/body blocks
  // and %init_val is the actual operand.
  SmallVector<OpAsmParser::UnresolvedOperand> operands;
  SmallVector<OpAsmParser::UnresolvedOperand> iterArgs;
  if (parser.parseLParen()) return failure();
  do {
    if (succeeded(parser.parseOptionalRParen())) break;
    OpAsmParser::UnresolvedOperand operand, iterArg;
    if (parser.parseOperand(iterArg) || parser.parseEqual() ||
        parser.parseOperand(operand))
      return failure();
    iterArgs.push_back(iterArg);
    operands.push_back(operand);
    if (succeeded(parser.parseOptionalRParen())) break;
    if (failed(parser.parseComma())) return failure();
  } while (true);
  if (!operands.empty()) {
    if (parser.parseColon() || parser.parseTypeList(result.types))
      return failure();
  }
  SmallVector<OpAsmParser::Argument> args;
  createArgs(iterArgs, result.types, args);
  if (parser.resolveOperands(operands, result.types, loc, result.operands) ||
      parser.parseOptionalAttrDictWithKeyword(result.attributes) ||
      parser.parseKeyword("cond") ||
      parser.parseRegion(*result.addRegion(), args) ||
      parser.parseKeyword("do") ||
      parser.parseRegion(*result.addRegion(), args))
    return failure();
  return success();
}

LogicalResult UniformDequantizeOp::inferReturnTypeComponents(
    MLIRContext*, std::optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  UniformDequantizeOp::Adaptor adaptor(operands, attributes, regions);
  return hlo::inferUniformDequantizeOp(location, adaptor.getOperand(),
                                       inferredReturnShapes);
}

}  // namespace stablehlo
}  // namespace mlir

using mlir::hlo::parseComplexOpType;
using mlir::hlo::parseCustomCallTarget;
using mlir::hlo::parseDenseI64Array;
using mlir::hlo::parseDotDimensionNumbers;
using mlir::hlo::parseExponentMantissa;
using mlir::hlo::parsePairwiseOpType;
using mlir::hlo::parseSameOperandsAndResultType;
using mlir::hlo::parseSelectOpType;
using mlir::hlo::parseTupleOpType;
using mlir::hlo::parseVariadicOperandWithAttribute;
using mlir::hlo::parseVariadicSameOperandsAndResultType;
using mlir::hlo::printComplexOpType;
using mlir::hlo::printCustomCallTarget;
using mlir::hlo::printDenseI64Array;
using mlir::hlo::printDotDimensionNumbers;
using mlir::hlo::printExponentMantissa;
using mlir::hlo::printPairwiseOpType;
using mlir::hlo::printSameOperandsAndResultType;
using mlir::hlo::printSelectOpType;
using mlir::hlo::printTupleOpType;
using mlir::hlo::printVariadicOperandWithAttribute;
using mlir::hlo::printVariadicSameOperandsAndResultType;

#define GET_OP_CLASSES
#include "stablehlo/dialect/StablehloOps.cpp.inc"

namespace mlir {
namespace stablehlo {

//===----------------------------------------------------------------------===//
// StableHLO Dialect Interfaces
//===----------------------------------------------------------------------===//

namespace {
struct StablehloDialectInlinerInterface : public DialectInlinerInterface {
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
  // Operations in StableHLO dialect are always legal to inline since they are
  // pure.
  bool isLegalToInline(Operation*, Region*, bool, IRMapping&) const final {
    return true;
  }
};

struct StablehloHloDialectInterface : public hlo::HloDialectInterface {
  using HloDialectInterface::HloDialectInterface;

  Type createTokenType() const override {
    return TokenType::get(getDialect()->getContext());
  }

  bool isTokenType(Type type) const override { return type.isa<TokenType>(); }

  Attribute createTypeExtensions(ArrayRef<int64_t> bounds) const override {
    return TypeExtensionsAttr::get(getDialect()->getContext(), bounds);
  }
};
}  // end anonymous namespace

//===----------------------------------------------------------------------===//
// StableHLO Dialect Constructor
//===----------------------------------------------------------------------===//

StablehloDialect::StablehloDialect(MLIRContext* context)
    : Dialect(getDialectNamespace(), context, TypeID::get<StablehloDialect>()) {
  addOperations<
#define GET_OP_LIST
#include "stablehlo/dialect/StablehloOps.cpp.inc"
      >();
  addInterfaces<StablehloDialectInlinerInterface>();
  addInterfaces<StablehloHloDialectInterface>();
  addBytecodeInterface(this);
  addTypes<TokenType>();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "stablehlo/dialect/StablehloAttrs.cpp.inc"
      >();
}

Type StablehloDialect::parseType(DialectAsmParser& parser) const {
  StringRef dataType;
  if (parser.parseKeyword(&dataType)) return Type();

  if (dataType == "token") return TokenType::get(getContext());
  parser.emitError(parser.getNameLoc())
      << "unknown StableHLO type: " << dataType;
  return nullptr;
}

void StablehloDialect::printType(Type type, DialectAsmPrinter& os) const {
  if (type.isa<TokenType>()) {
    os << "token";
    return;
  }
  os << "<unknown StableHLO type>";
}

// Entry point for Attribute parsing, TableGen generated code will handle the
// dispatch to the individual classes.
Attribute StablehloDialect::parseAttribute(DialectAsmParser& parser,
                                           Type type) const {
  StringRef attrTag;
  Attribute attr;
  auto parseResult = generatedAttributeParser(parser, &attrTag, type, attr);
  if (parseResult.has_value()) return attr;
  if (attrTag == "bounds")
    return hlo::parseTypeExtensions(
        // Casting to dialect interfaces doesn't work for const pointers,
        // so we have to cast away the constness of this.
        const_cast<StablehloDialect*>(this)
            ->getRegisteredInterface<hlo::HloDialectInterface>(),
        parser);
  parser.emitError(parser.getNameLoc(), "unknown StableHLO attribute");
  return Attribute();
}

// Entry point for Attribute printing, TableGen generated code will handle the
// dispatch to the individual classes.
void StablehloDialect::printAttribute(Attribute attr,
                                      DialectAsmPrinter& os) const {
  if (auto type_extensions = attr.dyn_cast<TypeExtensionsAttr>()) {
    hlo::printTypeExtensions(attr, os);
    return;
  }
  LogicalResult result = generatedAttributePrinter(attr, os);
  (void)result;
  assert(succeeded(result));
}

/// Helpers for attributes parsing.

static ParseResult parseDims(AsmParser& parser,
                             SmallVector<int64_t>& dimSizes) {
  dimSizes.clear();
  auto failOrDims = parseDimSizes(parser);
  if (failed(failOrDims)) return failure();
  dimSizes = std::move(*failOrDims);
  return success();
}

static ParseResult parseDimsWithMinimumElements(AsmParser& parser,
                                                SmallVector<int64_t>& dimSizes,
                                                int minElements) {
  if (failed(parseDims(parser, dimSizes))) return failure();
  if (static_cast<int64_t>(dimSizes.size()) < minElements)
    return parser.emitError(parser.getCurrentLocation())
           << "expected at least " << minElements << " element(s), found "
           << dimSizes.size();
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

// Custom printer and parser for ScatterDimensionNumbersAttr.
void ScatterDimensionNumbersAttr::print(AsmPrinter& printer) const {
  printStruct(printer, "scatter",
              std::make_pair("update_window_dims", getUpdateWindowDims()),
              std::make_pair("inserted_window_dims", getInsertedWindowDims()),
              std::make_pair("scatter_dims_to_operand_dims",
                             getScatterDimsToOperandDims()),
              std::make_pair("index_vector_dim", getIndexVectorDim()));
}
Attribute ScatterDimensionNumbersAttr::parse(AsmParser& parser, Type type) {
  if (failed(parser.parseLess())) return {};
  SmallVector<int64_t> updateWindowDims;
  SmallVector<int64_t> insertedWindowDims;
  SmallVector<int64_t> scatterDimsToOperandDims;
  int64_t indexVectorDim = 0;

  if (failed(parseStruct(
          parser,
          {"update_window_dims", "inserted_window_dims",
           "scatter_dims_to_operand_dims", "index_vector_dim"},
          {[&]() { return parseDims(parser, updateWindowDims); },
           [&]() { return parseDims(parser, insertedWindowDims); },
           [&]() { return parseDims(parser, scatterDimsToOperandDims); },
           [&]() { return parser.parseInteger(indexVectorDim); }}))) {
    parser.emitError(parser.getCurrentLocation())
        << "failed parsing scatter dimension numbers attribute";
    return {};
  }

  return ScatterDimensionNumbersAttr::get(
      parser.getContext(), updateWindowDims, insertedWindowDims,
      scatterDimsToOperandDims, indexVectorDim);
}

// Custom printer and parser for GatherDimensionNumbersAttr.
void GatherDimensionNumbersAttr::print(AsmPrinter& printer) const {
  printStruct(printer, "gather", std::make_pair("offset_dims", getOffsetDims()),
              std::make_pair("collapsed_slice_dims", getCollapsedSliceDims()),
              std::make_pair("start_index_map", getStartIndexMap()),
              std::make_pair("index_vector_dim", getIndexVectorDim()));
}

Attribute GatherDimensionNumbersAttr::parse(AsmParser& parser, Type type) {
  if (failed(parser.parseLess())) return {};

  SmallVector<int64_t> offsetDims;
  SmallVector<int64_t> collapsedSliceDims;
  SmallVector<int64_t> startIndexMap;
  int64_t indexVectorDim = 0;

  if (failed(parseStruct(
          parser,
          {"offset_dims", "collapsed_slice_dims", "start_index_map",
           "index_vector_dim"},
          {[&]() { return parseDims(parser, offsetDims); },
           [&]() { return parseDims(parser, collapsedSliceDims); },
           [&]() { return parseDims(parser, startIndexMap); },
           [&]() { return parser.parseInteger(indexVectorDim); }}))) {
    parser.emitError(parser.getCurrentLocation())
        << "failed parsing gather dimension numbers attribute";
    return {};
  }

  return GatherDimensionNumbersAttr::get(parser.getContext(), offsetDims,
                                         collapsedSliceDims, startIndexMap,
                                         indexVectorDim);
}

// Custom printer and parser for DotDimensionNumbersAttr.
void DotDimensionNumbersAttr::print(AsmPrinter& printer) const {
  printStruct(
      printer, "dot",
      std::make_pair("lhs_batching_dimensions", getLhsBatchingDimensions()),
      std::make_pair("rhs_batching_dimensions", getRhsBatchingDimensions()),
      std::make_pair("lhs_contracting_dimensions",
                     getLhsContractingDimensions()),
      std::make_pair("rhs_contracting_dimensions",
                     getRhsContractingDimensions()));
}

Attribute DotDimensionNumbersAttr::parse(AsmParser& parser, Type type) {
  if (failed(parser.parseLess())) return {};

  SmallVector<int64_t> lhsBatchingDimensions;
  SmallVector<int64_t> rhsBatchingDimensions;
  SmallVector<int64_t> lhsContractingDimensions;
  SmallVector<int64_t> rhsContractingDimensions;

  if (failed(parseStruct(
          parser,
          {"lhs_batching_dimensions", "rhs_batching_dimensions",
           "lhs_contracting_dimensions", "rhs_contracting_dimensions"},
          {[&]() { return parseDims(parser, lhsBatchingDimensions); },
           [&]() { return parseDims(parser, rhsBatchingDimensions); },
           [&]() { return parseDims(parser, lhsContractingDimensions); },
           [&]() { return parseDims(parser, rhsContractingDimensions); }}))) {
    parser.emitError(parser.getCurrentLocation())
        << "failed parsing dot dimension numbers attribute";
    return {};
  }
  return DotDimensionNumbersAttr::get(
      parser.getContext(), lhsBatchingDimensions, rhsBatchingDimensions,
      lhsContractingDimensions, rhsContractingDimensions);
}

namespace {
enum NonSpatialDim : int64_t {
  IOBatch = -1,    // Input or output batch dimension
  IOFeature = -2,  // Input or output feature dimension
  KIFeature = -3,  // Kernel input feature dimension
  KOFeature = -4,  // Kernel output feature dimensions.
};

struct DenseMapInfoNonSpatialDim {
  static inline NonSpatialDim getEmptyKey() {
    return NonSpatialDim(DenseMapInfo<int64_t>::getEmptyKey());
  }

  static inline NonSpatialDim getTombstoneKey() {
    return NonSpatialDim(DenseMapInfo<int64_t>::getTombstoneKey());
  }

  static unsigned getHashValue(const NonSpatialDim& key) {
    return DenseMapInfo<int64_t>::getHashValue(key);
  }

  static bool isEqual(const NonSpatialDim& lhs, const NonSpatialDim& rhs) {
    return lhs == rhs;
  }
};

char nonSpatialDimToString(NonSpatialDim dim) {
  switch (dim) {
    case IOBatch:
      return 'b';
    case IOFeature:
      return 'f';
    case KIFeature:
      return 'i';
    case KOFeature:
      return 'o';
  }
  llvm::report_fatal_error("unsupported NonSpatialDim");
}
}  // namespace

// Custom printer and parser for convolution attribute.
void printConvolutionDimensions(AsmPrinter& p, ConvDimensionNumbersAttr dnums) {
  // TODO(b/202040055): we should check the attribute invariant and print the
  // "raw" form if they are violated, otherwise we'll crash here.
  auto printDim =
      [&p](ArrayRef<int64_t> spatialDims,
           ArrayRef<std::pair<int64_t, NonSpatialDim>> non_spatialDims) {
        llvm::SmallVector<int64_t> dims(non_spatialDims.size() +
                                        spatialDims.size());
        // Fill each element of dims with a (< 0) NonSpatialDim enum or a (>=0)
        // spatial dimension index.
        for (const std::pair<int64_t, NonSpatialDim>& nonSpatialDim :
             non_spatialDims)
          dims[nonSpatialDim.first] = nonSpatialDim.second;
        for (const auto& spatial_dim : llvm::enumerate(spatialDims))
          dims[spatial_dim.value()] = static_cast<int64_t>(spatial_dim.index());

        // Each dimension numbers will be printed as a comma separated list
        // surrounded by square brackets, e.g., [b, 0, 1, 2, f]
        p << '[';
        llvm::interleaveComma(dims, p, [&](int64_t dim) {
          if (dim >= 0)
            p << dim;
          else
            p << nonSpatialDimToString(static_cast<NonSpatialDim>(dim));
        });
        p << ']';
      };

  printDim(dnums.getInputSpatialDimensions(),
           {{dnums.getInputBatchDimension(), IOBatch},
            {dnums.getInputFeatureDimension(), IOFeature}});
  p << "x";
  printDim(dnums.getKernelSpatialDimensions(),
           {{dnums.getKernelInputFeatureDimension(), KIFeature},
            {dnums.getKernelOutputFeatureDimension(), KOFeature}});
  p << "->";
  printDim(dnums.getOutputSpatialDimensions(),
           {{dnums.getOutputBatchDimension(), IOBatch},
            {dnums.getOutputFeatureDimension(), IOFeature}});
}

void printConvolutionDimensions(AsmPrinter& p, Operation*,
                                ConvDimensionNumbersAttr dnums) {
  printConvolutionDimensions(p, dnums);
}

// Custom printer and parser for ConvDimensionNumbersAttr.
void ConvDimensionNumbersAttr::print(AsmPrinter& printer) const {
  printer << "<";
  printConvolutionDimensions(printer, *this);
  printer << ">";
}

// If the attribute is written with `#stablehlo.conv raw<`, we parse it as
// a struct instead of the compressed format. This enables writing tests
// covering impossible/invalid internal representation for the attribute.
static ParseResult parseConvolutionDimensionsRaw(
    AsmParser& parser, ConvDimensionNumbersAttr& dnums) {
  int64_t inputBatchDimension = 0;
  int64_t inputFeatureDimension = 0;
  SmallVector<int64_t> inputSpatialDimensions;
  int64_t kernelInputFeatureDimension = 0;
  int64_t kernelOutputFeatureDimension = 0;
  SmallVector<int64_t> kernelSpatialDimensions;
  int64_t outBatchDimension = 0;
  int64_t outputFeatureDimension = 0;
  SmallVector<int64_t> outputSpatialDimensions;
  if (failed(parseStruct(
          parser,
          {"input_batch_dimension", "input_feature_dimension",
           "input_spatial_dimensions", "kernel_input_feature_dimension",
           "kernel_output_feature_dimension", "kernel_spatial_dimensions",
           "output_batch_dimension", "output_feature_dimension",
           "output_spatial_dimensions"},
          {
              [&]() { return parser.parseInteger(inputBatchDimension); },
              [&]() { return parser.parseInteger(inputFeatureDimension); },
              [&]() { return parseDims(parser, inputSpatialDimensions); },
              [&]() {
                return parser.parseInteger(kernelInputFeatureDimension);
              },
              [&]() {
                return parser.parseInteger(kernelOutputFeatureDimension);
              },
              [&]() { return parseDims(parser, kernelSpatialDimensions); },
              [&]() { return parser.parseInteger(outBatchDimension); },
              [&]() { return parser.parseInteger(outputFeatureDimension); },
              [&]() { return parseDims(parser, outputSpatialDimensions); },
          }))) {
    parser.emitError(parser.getCurrentLocation())
        << "failed parsing dot dimension numbers attribute";
    return failure();
  }
  dnums = ConvDimensionNumbersAttr::get(
      parser.getBuilder().getContext(), inputBatchDimension,
      inputFeatureDimension, inputSpatialDimensions,
      kernelInputFeatureDimension, kernelOutputFeatureDimension,
      kernelSpatialDimensions, outBatchDimension, outputFeatureDimension,
      outputSpatialDimensions);
  return success();
}

ParseResult parseConvolutionDimensions(AsmParser& parser,
                                       ConvDimensionNumbersAttr& dnums) {
  // Parsing a single set of dim numbers gives the spatial dimensions as a
  // single ArrayRef<int64_t> and a list of non-spatial dimensions as
  // IntegerAttrs (indexed by the NonSpatialDim enum).
  using parse_dim_result_t =
      std::pair<llvm::SmallVector<int64_t>,
                llvm::SmallDenseMap<NonSpatialDim, int64_t, 4,
                                    DenseMapInfoNonSpatialDim>>;

  // Note that the allowedNonSpatialDims is a set (as opposed to unordered
  // set) because its used to print a list of allowed non spatial dims in the
  // error messages, so making it a set keeps the error messages deterministic.
  auto parseDims =
      [&](std::set<NonSpatialDim, std::greater<>> allowedNonSpatialDims,
          parse_dim_result_t& parsedDims) -> ParseResult {
    auto& spatialDims = std::get<0>(parsedDims);
    auto& nonSpatialDims = std::get<1>(parsedDims);
    spatialDims.clear();
    nonSpatialDims.clear();

    // Parse the starting [
    if (parser.parseLSquare()) return failure();

    llvm::SmallDenseMap<int64_t, int64_t> spatialDimsMap;
    constexpr int64_t kInvalidDimension = -1;
    // Keep track of the maximum spatial dimension parsed as we expect to see
    // all the dimensions from 0 to maximum dimension parsed.
    int64_t maxParsedSpatialDim = kInvalidDimension;

    int64_t index = 0;
    do {
      int64_t spatialDim;
      auto dimLocation = parser.getCurrentLocation();
      OptionalParseResult parseResult = parser.parseOptionalInteger(spatialDim);
      if (parseResult.has_value()) {
        if (parseResult.value().failed()) return failure();
        // We were successful in parsing an integer. Check if it is a valid
        // dimension (non-negative and no duplicate) and add its index to the
        // spatial dims map.
        if (spatialDim < 0)
          return parser.emitError(dimLocation)
                 << "Unexpected dimension " << spatialDim;
        if (!spatialDimsMap
                 .insert(std::pair<int64_t, int64_t>(spatialDim, index))
                 .second)
          return parser.emitError(dimLocation)
                 << "Duplicate entries for spatial dimension " << spatialDim;
        maxParsedSpatialDim = std::max(spatialDim, maxParsedSpatialDim);
      } else {
        // We did not parse an integer. We expect a keyword token.
        StringRef keyword;
        if (parser.parseKeyword(&keyword)) return failure();
        if (keyword.size() != 1 || allowedNonSpatialDims.empty())
          return parser.emitError(dimLocation, "Unexpected keyword ")
                 << keyword;
        // Check if the keyword matches one of the allowed non-spatial dims.
        // If so, add it to the non_spatial dims and remove it from the
        // allowed set so that it won't be allowed again.
        bool isAllowed = false;
        for (NonSpatialDim allowed : allowedNonSpatialDims) {
          if (keyword[0] == nonSpatialDimToString(allowed)) {
            nonSpatialDims.insert({allowed, index});
            allowedNonSpatialDims.erase(allowed);
            isAllowed = true;
            break;
          }
        }

        if (!isAllowed) {
          mlir::InFlightDiagnostic diag =
              parser.emitError(dimLocation, "Unexpected dimension ");
          diag << keyword << ", expecting ";
          llvm::interleaveComma(
              allowedNonSpatialDims, diag,
              [&](NonSpatialDim dim) { diag << nonSpatialDimToString(dim); });
          return diag;
        }
      }
      index++;
    } while (parser.parseOptionalComma().succeeded());

    // Make sure all expected non-spatial dimensions are parsed.
    if (!allowedNonSpatialDims.empty()) {
      mlir::InFlightDiagnostic diag =
          parser.emitError(parser.getCurrentLocation(), "Expected dimensions ");
      llvm::interleaveComma(
          allowedNonSpatialDims, diag,
          [&](NonSpatialDim dim) { diag << nonSpatialDimToString(dim); });
      diag << " not specified";
      return diag;
    }

    // parse ending ]
    if (parser.parseRSquare()) return failure();

    // Number of expected spatial dimensions is one more than the maximum parsed
    // spatial dimension. For example, if we parse [0, 3, 2, b, i, 1], then the
    // maximum parsed spatial dimension is 3 and the number of expected spatial
    // dimensions is 4.
    int64_t numSpatialDimensions = maxParsedSpatialDim + 1;
    spatialDims.resize(numSpatialDimensions);
    // Store spatial dimensions in a vector which maps spatial dim (vector
    // index) -> index in the tensor dimensions. For example, for parsed
    // dimension numbers [0, 3, 2, b, i, 1] the spatial dimension vector would
    // be [0, 5, 2, 1].
    //
    // Get all the unspecified spatial dimensions to throw a more descriptive
    // error later.
    llvm::SmallVector<int64_t> unspecifiedSpatialDims;
    constexpr int kPrintUnspecifiedDimsMax = 10;
    for (int dim = 0; dim < numSpatialDimensions; ++dim) {
      auto it = spatialDimsMap.find(dim);
      if (it == spatialDimsMap.end()) {
        // Have an upper bound on the number of unspecified dimensions to print
        // in the error message.
        if (unspecifiedSpatialDims.size() < kPrintUnspecifiedDimsMax)
          unspecifiedSpatialDims.push_back(dim);
        continue;
      }
      spatialDims[dim] = it->second;
    }

    // Verify that we got all spatial dimensions between 0 and maximum parsed
    // spatial dimension.
    if (!unspecifiedSpatialDims.empty()) {
      mlir::InFlightDiagnostic diag = parser.emitError(
          parser.getCurrentLocation(), "Expected spatial dimensions ");
      llvm::interleaveComma(unspecifiedSpatialDims, diag);
      diag << " not specified";
      return diag;
    }

    return success();
  };

  parse_dim_result_t parsedDims;
  if (parseDims({IOBatch, IOFeature}, parsedDims)) return failure();

  llvm::SmallVector<int64_t> inputSpatialDimensions = parsedDims.first;
  int64_t inputBatchDimension = parsedDims.second[IOBatch];
  int64_t inputFeatureDimension = parsedDims.second[IOFeature];

  if (parser.parseKeyword("x")) return failure();
  if (parseDims({KIFeature, KOFeature}, parsedDims)) return failure();

  llvm::SmallVector<int64_t> kernelSpatialDimensions = parsedDims.first;
  int64_t kernelInputFeatureDimension = parsedDims.second[KIFeature];
  int64_t kernelOutputFeatureDimension = parsedDims.second[KOFeature];

  if (parser.parseArrow()) return failure();
  if (parseDims({IOBatch, IOFeature}, parsedDims)) return failure();

  llvm::SmallVector<int64_t> outputSpatialDimensions = parsedDims.first;
  const int64_t outBatchDimension = parsedDims.second[IOBatch];
  const int64_t outputFeatureDimension = parsedDims.second[IOFeature];
  dnums = ConvDimensionNumbersAttr::get(
      parser.getBuilder().getContext(), inputBatchDimension,
      inputFeatureDimension, inputSpatialDimensions,
      kernelInputFeatureDimension, kernelOutputFeatureDimension,
      kernelSpatialDimensions, outBatchDimension, outputFeatureDimension,
      outputSpatialDimensions);

  return success();
}

Attribute ConvDimensionNumbersAttr::parse(AsmParser& parser, Type type) {
  if (failed(parser.parseLess())) return {};
  ConvDimensionNumbersAttr dnums;
  if (succeeded(parser.parseOptionalKeyword("raw"))) {
    if (failed(parseConvolutionDimensionsRaw(parser, dnums))) return {};
    return dnums;
  }
  if (failed(parseConvolutionDimensions(parser, dnums))) return {};
  if (failed(parser.parseGreater())) return {};
  return dnums;
}

// Custom printer and parser for ArgResultAliasAttr.
constexpr char kMustAlias[] = "must_alias";
constexpr char kResult[] = "result_index";
constexpr char kArgTupleIndices[] = "tuple_indices";

void ArgResultAliasAttr::print(AsmPrinter& printer) const {
  printer << "<";

  // The attribute can have empty tuple indices. Only print argument tuple
  // indices if they are non-empty.
  if (!getArgTupleIndices().empty())
    printer << kArgTupleIndices << " = [" << getArgTupleIndices() << "], ";

  // Print the result index followed by any result tuple indices if present.
  printer << kResult << " = [";
  printer << getResultIndex();
  if (!getResultTupleIndices().empty())
    printer << ", " << getResultTupleIndices();
  printer << "]";

  // Print the "must_alias" keyword if this is a must alias, otherwise skip.
  if (getIsMustAlias()) printer << ", " << kMustAlias;

  printer << ">";
}

Attribute ArgResultAliasAttr::parse(AsmParser& parser, Type type) {
  if (failed(parser.parseLess())) return {};
  llvm::SmallVector<int64_t> argTupleIndices;
  // The first element of result indices holds the aliased result index and the
  // remaining elements are the result tuple indices.
  llvm::SmallVector<int64_t> resultIndices;
  bool isMustAlias = false;

  // This conveys to parseStruct that keyword "must_alias" (3rd field) is not
  // followed by a "=", but other fields are.
  llvm::SmallVector<bool, 3> parseEqual = {true, true, false};

  if (failed(parseStruct(parser, {kArgTupleIndices, kResult, kMustAlias},
                         {[&]() { return parseDims(parser, argTupleIndices); },
                          [&]() {
                            // Since the first element is the index of result,
                            // at least one element is expected.
                            return parseDimsWithMinimumElements(
                                parser, resultIndices, /*minElements=*/1);
                          },
                          [&]() {
                            // always succeeds if the keyword "must_alias" was
                            // parsed
                            isMustAlias = true;
                            return success();
                          }},
                         parseEqual))) {
    parser.emitError(parser.getCurrentLocation())
        << "failed parsing argument-result alias attribute";
    return {};
  }

  int64_t resultIndex = resultIndices[0];
  auto resultTupleIndices =
      ArrayRef<int64_t>{resultIndices.begin() + 1, resultIndices.end()};

  return ArgResultAliasAttr::get(parser.getContext(), argTupleIndices,
                                 resultIndex, resultTupleIndices, isMustAlias);
}

// Returns the element type pointed to by `indices` in type `t`. If the indices
// are invalid, returns nullptr.
static Type getTypeFromTupleIndices(Type type, ArrayRef<int64_t> indices) {
  Type current = type;
  for (auto index : indices) {
    TupleType tupleType = current.dyn_cast<TupleType>();
    if (!tupleType || index >= static_cast<int64_t>(tupleType.size()))
      return {};
    current = tupleType.getType(index);
  }
  return current;
}

static LogicalResult verifyArgResultAliasAttr(StringAttr attrName,
                                              ArgResultAliasAttr aliasAttr,
                                              unsigned argIndex,
                                              Operation* op) {
  // The attribute can only be applied to function-like operations.
  if (!isa<mlir::FunctionOpInterface>(op))
    return op->emitOpError() << "attribute " << attrName
                             << " can only be used on function-like operations";

  // Verify there are no negative indices.
  auto tupleIndices = llvm::concat<const int64_t>(
      aliasAttr.getArgTupleIndices(), aliasAttr.getResultTupleIndices());
  if (llvm::any_of(tupleIndices, [](const int64_t val) { return val < 0; }) ||
      aliasAttr.getResultIndex() < 0)
    return op->emitOpError()
           << "attribute " << attrName
           << " expects all argument and result indices to be >= 0";

  // Verify that the result index is not out of range. Since the attribute is a
  // function argument attribute, the argument index is always correct when this
  // verifier is called.
  FunctionOpInterface funcOp = cast<FunctionOpInterface>(op);
  ArrayRef<Type> argTypes = funcOp.getArgumentTypes();
  ArrayRef<Type> resultTypes = funcOp.getResultTypes();
  if (aliasAttr.getResultIndex() >= static_cast<int64_t>(resultTypes.size()))
    return op->emitOpError()
           << "attribute " << attrName
           << " result index is out of range, must be <" << resultTypes.size();

  // Verify that argument and result types pointed to by the indices are valid
  // and compatible.
  Type argType = getTypeFromTupleIndices(argTypes[argIndex],
                                         aliasAttr.getArgTupleIndices());
  if (!argType)
    return op->emitOpError()
           << "attribute " << attrName << " argument tuple indices are invalid";
  Type resultType =
      getTypeFromTupleIndices(resultTypes[aliasAttr.getResultIndex()],
                              aliasAttr.getResultTupleIndices());
  if (!resultType)
    return op->emitOpError()
           << "attribute " << attrName << " result tuple indices are invalid";

  if (failed(mlir::verifyCompatibleShape(argType, resultType)) ||
      getElementTypeOrSelf(argType) != getElementTypeOrSelf(resultType))
    return op->emitOpError() << "attribute " << attrName
                             << " aliases do not have compatible types, "
                             << argType << " vs. " << resultType;
  return success();
}

namespace {
// Custom formatting for convolution window attributes.
void printWindowAttribute(OpAsmPrinter& p, DenseElementsAttr attribute) {
  if (attribute.getElementType().isInteger(/*width=*/1)) {
    // boolean attribute.
    llvm::interleaveComma(attribute.getValues<bool>(), p,
                          [&](bool b) { p << (b ? 1 : 0); });
    return;
  }
  if (attribute.getType().getRank() == 2) {
    // Padding is Nx2 attribute.
    auto it = attribute.value_begin<int64_t>();
    std::vector<std::pair<int64_t, int64_t>> values(attribute.getNumElements() /
                                                    2);
    for (auto& item : values) {
      int64_t first = *it;
      ++it;
      int64_t second = *it;
      ++it;
      item = {first, second};
    }
    llvm::interleaveComma(
        values, p, [&](const std::pair<int64_t, int64_t> pair) {
          p << '[' << pair.first << ", " << pair.second << ']';
        });
  } else {
    llvm::interleaveComma(attribute.getValues<int64_t>(), p);
  }
}
}  // namespace

void printWindowAttributes(OpAsmPrinter& p, Operation* /*op*/,
                           std::optional<DenseIntElementsAttr> windowStrides,
                           std::optional<DenseIntElementsAttr> padding,
                           std::optional<DenseIntElementsAttr> lhsDilation,
                           std::optional<DenseIntElementsAttr> rhsDilation,
                           std::optional<DenseElementsAttr> windowReversal) {
  using pair_t = std::pair<DenseElementsAttr, StringRef>;
  std::array<pair_t, 5> printedAttributes = {{
      {windowStrides ? *windowStrides : nullptr, "stride"},
      {padding ? *padding : nullptr, "pad"},
      {lhsDilation ? *lhsDilation : nullptr, "lhs_dilate"},
      {rhsDilation ? *rhsDilation : nullptr, "rhs_dilate"},
      {windowReversal ? *windowReversal : nullptr, "reverse"},
  }};

  // Do not print attributes that do no exist.
  auto nonNullAttributes = llvm::make_filter_range(
      printedAttributes,
      [](const pair_t& a) { return static_cast<bool>(a.first); });

  llvm::interleaveComma(nonNullAttributes, p, [&](const pair_t& a) {
    p << a.second << " = [";
    printWindowAttribute(p, a.first);
    p << "]";
  });
}

ParseResult parseWindowAttributes(OpAsmParser& parser,
                                  DenseIntElementsAttr& windowStrides,
                                  DenseIntElementsAttr& padding,
                                  DenseIntElementsAttr& lhsDilation,
                                  DenseIntElementsAttr& rhsDilation,
                                  DenseElementsAttr& windowReversal) {
  StringRef attributeName;

  llvm::StringSet<> allowedAttributeNames{
      {"stride", "pad", "lhs_dilate", "rhs_dilate", "reverse"}};

  while (parser.parseOptionalKeyword(&attributeName).succeeded()) {
    // Verify that the attribute name is valid and erase it.
    if (!allowedAttributeNames.erase(attributeName))
      return parser.emitError(parser.getCurrentLocation(),
                              "Unexpected keyword ")
             << attributeName;

    if (parser.parseEqual()) return failure();

    // parse the attribute value. We need to support either 1D and Nx2 array of
    // integers to parse.
    llvm::SmallVector<int64_t> values;
    auto int64Parser = [&]() {
      return parser.parseInteger(values.emplace_back(0));
    };

    if (attributeName == "pad") {
      // Parse 2D array of integers.
      // Helper to parse an array of two integer elements such as [e0, e1].
      auto innerParser = [&]() -> ParseResult {
        size_t numOldElements = values.size();
        if (parser.parseCommaSeparatedList(mlir::AsmParser::Delimiter::Square,
                                           int64Parser))
          return failure();
        size_t numParsedElements = values.size() - numOldElements;
        constexpr size_t kExpectedElements = 2;
        if (numParsedElements != kExpectedElements)
          return parser.emitError(parser.getCurrentLocation())
                 << "Expected array with " << kExpectedElements
                 << " elements, got " << numParsedElements
                 << " elements instead";
        return success();
      };

      if (parser.parseCommaSeparatedList(AsmParser::Delimiter::Square,
                                         innerParser))
        return failure();
      const int64_t size = static_cast<int64_t>(values.size());
      // values should be filled with the Nx2 padding values.
      assert(size % 2 == 0);
      auto ty = RankedTensorType::get({size / 2, 2},
                                      parser.getBuilder().getIntegerType(64));
      padding = DenseIntElementsAttr::get(ty, values);
    } else {
      // Parse 1D array of integers.
      if (parser.parseCommaSeparatedList(AsmParser::Delimiter::Square,
                                         int64Parser))
        return failure();
      const int64_t size = static_cast<int64_t>(values.size());
      if (attributeName == "reverse") {
        auto ty = RankedTensorType::get({size},
                                        parser.getBuilder().getIntegerType(1));
        auto boolVector = llvm::to_vector<4>(
            llvm::map_range(values, [](int64_t v) { return v != 0; }));
        windowReversal = DenseElementsAttr::get(ty, boolVector);
      } else {
        auto attr = parser.getBuilder().getI64TensorAttr(values);

        if (attributeName == "stride") {
          windowStrides = attr;
        } else if (attributeName == "lhs_dilate") {
          lhsDilation = attr;
        } else if (attributeName == "rhs_dilate") {
          rhsDilation = attr;
        } else {
          llvm::report_fatal_error("unsupported attribute name");
        }
      }
    }
    // continue parsing if there is a comma at the end.
    if (parser.parseOptionalComma().failed()) break;
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Builder utilities
//===----------------------------------------------------------------------===//

// Builds the region `body` for stablehlo.sort's comparator: for each type in
// `element_types`, create two block arguments, one for lhs and one for rhs, and
// generates stablehlo.compare op to compare them with the given `direction`.
//
// Note that this right now only does comparision on the first pair of block
// arguments.
static void buildSortComparisonBody(llvm::ArrayRef<Type> elementTypes,
                                    ComparisonDirection direction,
                                    std::optional<StringRef> compareType,
                                    Region* body, OpBuilder* builder) {
  OpBuilder::InsertionGuard insertionPointGurad(*builder);

  Location loc = body->getLoc();
  Block* block = builder->createBlock(body);
  // Add two arguments for each element type.
  for (Type elementType : elementTypes) {
    TensorType tensorType = RankedTensorType::get({}, elementType);
    block->addArguments({tensorType, tensorType},
                        SmallVector<Location, 2>(2, loc));
  }

  ComparisonType typeAttr;
  if (compareType)
    typeAttr = symbolizeComparisonType(*compareType).value();
  else
    typeAttr = ComparisonType::NOTYPE;
  Value compare = builder->create<CompareOp>(
      loc, block->getArgument(0), block->getArgument(1), direction, typeAttr);

  builder->create<ReturnOp>(loc, compare);
}

SortOp createSortOp(PatternRewriter* rewriter, const Location& loc,
                    const llvm::ArrayRef<Value>& operands,
                    const llvm::ArrayRef<Type>& elementTypes, int64_t dimension,
                    bool isStable, ComparisonDirection direction) {
  assert(!operands.empty() && "No operands to sort");
  // Create the sort op.
  auto sortOp = rewriter->create<SortOp>(loc, operands, dimension, isStable);

  // Use TOTALORDER comparison type instead of the default comparison if the
  // element type is of type float.
  std::optional<StringRef> compareType = std::nullopt;
  for (const auto& elementType : elementTypes) {
    if (elementType.isa<FloatType>()) {
      compareType.emplace("TOTALORDER");
      break;
    }
  }
  buildSortComparisonBody(elementTypes, direction, compareType,
                          &sortOp.getComparator(), rewriter);
  return sortOp;
}

//===----------------------------------------------------------------------===//
// StableHLO Dialect Hooks
//===----------------------------------------------------------------------===//

Operation* StablehloDialect::materializeConstant(OpBuilder& builder,
                                                 Attribute value, Type type,
                                                 Location loc) {
  auto elementsAttr = value.dyn_cast<ElementsAttr>();
  // HLO dialect constants only support ElementsAttr unlike standard dialect
  // constant which supports all attributes.
  if (!elementsAttr) return nullptr;
  // HLO dialect constants require the type of value and result to match.
  if (type != elementsAttr.getType()) return nullptr;

  return builder.create<ConstantOp>(loc, type, elementsAttr);
}

LogicalResult StablehloDialect::verifyRegionArgAttribute(
    Operation* op, unsigned /*regionIndex*/, unsigned argIndex,
    NamedAttribute attr) {
  if (auto aliasAttr = attr.getValue().dyn_cast<ArgResultAliasAttr>())
    if (failed(
            verifyArgResultAliasAttr(attr.getName(), aliasAttr, argIndex, op)))
      return failure();
  return success();
}

LogicalResult StablehloDialect::verifyOperationAttribute(Operation* op,
                                                         NamedAttribute attr) {
  if (auto aliasAttr = attr.getValue().dyn_cast<ArgResultAliasAttr>())
    if (!isa<mlir::FunctionOpInterface>(op))
      return op->emitOpError()
             << "attribute " << attr.getName()
             << " can only be used on function-like operations";
  return success();
}

}  // namespace stablehlo
}  // namespace mlir
