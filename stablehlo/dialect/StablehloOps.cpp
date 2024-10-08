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
#include <iterator>
#include <numeric>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/Regex.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/TypeID.h"
#include "mlir/Transforms/InliningUtils.h"
#include "stablehlo/dialect/AssemblyFormat.h"
#include "stablehlo/dialect/Base.h"
#include "stablehlo/dialect/StablehloBytecode.h"
#include "stablehlo/dialect/StablehloOps.h.inc"
#include "stablehlo/dialect/TypeInference.h"

// Include order matters
#define GET_TYPEDEF_CLASSES
#include "stablehlo/dialect/StablehloTypeDefs.cpp.inc"
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
// ReduceScatterOp
//===----------------------------------------------------------------------===//

LogicalResult ReduceScatterOp::verify() {
  int64_t channelId = 0;
  if (auto channelHandleAttr = getChannelHandleAttr())
    channelId = channelHandleAttr.getHandle();

  return hlo::verifyReduceScatterOp(
      getLoc(), getOperand(), getScatterDimension(), getReplicaGroups(),
      channelId, getUseGlobalDeviceIds(), getComputation(), getResult());
}

mlir::Speculation::Speculatability ReduceScatterOp::getSpeculatability() {
  auto inputType = getOperand().getType();
  auto resultType = getResult().getType();
  auto scatterDim = getScatterDimension();
  // The actual size of the `scatterDim` depends on the number of processes,
  // which is only known at runtime. If it is dynamic, there is no expectation,
  // so there cannot be a mismatch. If it is static, the actual number may
  // differ at runtime, leading to UB. See scatter_c8 in the spec.
  if (!resultType.isDynamicDim(scatterDim))
    return mlir::Speculation::NotSpeculatable;
  for (size_t i : llvm::seq(resultType.getRank())) {
    if (i == scatterDim) continue;
    if (!resultType.isDynamicDim(i) && inputType.isDynamicDim(i))
      return mlir::Speculation::NotSpeculatable;
  }
  return mlir::Speculation::Speculatable;
}

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

INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(AndOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(Atan2Op)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(CbrtOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(CeilOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(ClzOp)
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
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(TanOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(TanhOp)
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(XorOp)

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

LogicalResult AddOp::inferReturnTypeComponents(
    MLIRContext* context, std::optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes,
    OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  SmallVector<Type> inferredReturnTypes;
  if (failed(inferReturnTypes(context, location, operands.getValues(),
                              attributes, properties, regions,
                              inferredReturnTypes)))
    return failure();
  if (inferredReturnTypes.size() != 1) return failure();
  auto inferredReturnType = dyn_cast<ShapedType>(inferredReturnTypes[0]);
  if (!inferredReturnType) return failure();
  inferredReturnShapes.push_back(inferredReturnType);
  return success();
}

LogicalResult AddOp::verify() {
  return hlo::verifyAddOp(getLoc(), getOperation(), getLhs().getType(),
                          getRhs().getType(), getResult().getType());
}

//===----------------------------------------------------------------------===//
// AfterAllOp
//===----------------------------------------------------------------------===//

LogicalResult AfterAllOp::inferReturnTypes(
    MLIRContext* context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  return hlo::inferAfterAllOp(getStablehloDialect(context), location,
                              inferredReturnTypes);
}

//===----------------------------------------------------------------------===//
// CompositeOp
//===----------------------------------------------------------------------===//

LogicalResult CompositeOp::verifySymbolUses(
    SymbolTableCollection& symbolTable) {
  return hlo::verifyCompositeOp(getLoc(), getOperation(), getName(),
                                getDecomposition(), symbolTable);
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

void ConstantOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  mlir::TensorType type = getType();
  if (isa<IntegerType>(type.getElementType())) {
    setNameFn(getResult(), "c");
  } else {
    setNameFn(getResult(), "cst");
  }
}

OpFoldResult ConstantOp::fold(FoldAdaptor adaptor) {
  assert(adaptor.getOperands().empty() && "constant has no operands");

  // Return the held attribute value.
  return getValue();
}

// Builds a constant op with the specified attribute `value`.
void ConstantOp::build(OpBuilder& /*builder*/, OperationState& result,
                       Attribute value) {
  ShapedType type;
  if (auto elemAttr = dyn_cast<ElementsAttr>(value)) {
    type = cast<ShapedType>(elemAttr.getType());
  } else if (isa<BoolAttr, FloatAttr, IntegerAttr>(value)) {
    // All XLA types must be tensor types. In the build() method, we want to
    // provide more flexibility by allowing attributes of scalar types. But we
    // need to wrap it up with ElementsAttr to construct valid XLA constants.
    type =
        RankedTensorType::get(/*shape=*/{}, cast<TypedAttr>(value).getType());
    value = DenseElementsAttr::get(type, value);
  } else if (auto complexAttr = dyn_cast<complex::NumberAttr>(value)) {
    type = RankedTensorType::get(/*shape=*/{},
                                 cast<TypedAttr>(complexAttr).getType());
    value = DenseElementsAttr::get(type, complexAttr.getValue());
  }

  // TODO: support other XLA specific types.
  assert(type && "unsupported attribute type for building constant");
  result.types.push_back(type);
  result.addAttribute("value", value);
}

LogicalResult ConstantOp::inferReturnTypes(
    MLIRContext*, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  ConstantOpAdaptor adaptor(operands, attributes, properties);
  return hlo::inferConstantOp(location, adaptor.getValue(),
                              inferredReturnTypes);
}

bool ConstantOp::isCompatibleReturnTypes(TypeRange l, TypeRange r) {
  if (l.size() != r.size() || l.size() != 1) return false;
  auto lhsTy = dyn_cast<ShapedType>(l.front());
  auto rhsTy = dyn_cast<ShapedType>(r.front());
  if (!lhsTy || !rhsTy) return false;
  // For comparisons of the uniform quantized element based tensor type, use the
  // storage type since the constant value will be stored through the underlying
  // storage type.
  if (auto rhsElemTy = dyn_cast<quant::QuantizedType>(rhsTy.getElementType()))
    rhsTy = hlo::getSameShapeTensorType(rhsTy, rhsElemTy.getStorageType());
  return lhsTy == rhsTy;
}

ParseResult ConstantOp::parse(OpAsmParser& parser, OperationState& result) {
  return hlo::parseConstantOp(parser, result);
}

void ConstantOp::print(::mlir::OpAsmPrinter& p) {
  hlo::printConstantOp(p, getOperation(), getValue());
}

//===----------------------------------------------------------------------===//
// CreateTokenOp
//===----------------------------------------------------------------------===//

LogicalResult CreateTokenOp::inferReturnTypes(
    MLIRContext* context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
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
        auto layout = cast<DenseIntElementsAttr>(
            std::get<1>(indexedTypeAndLayout.value()));

        if (isa<TupleType>(type))
          return emitOpError() << "Tuple types are not fully supported with "
                                  "layout constraints yet";
        auto shapedType = dyn_cast<ShapedType>(type);

        // For non-tensor types such as !stablehlo.token, the layout should be
        // empty.
        if (!shapedType) {
          if (layout.empty()) continue;
          return emitOpError()
                 << "Only tensor types can have non-empty layout: " << valueName
                 << " #" << index << " of type " << type << " has layout "
                 << layout;
        }

        // For unranked tensors, we cannot verify the compatibility with layout
        // any further.
        if (!shapedType.hasRank()) continue;

        // Layout must be a permutation of [0, N) where N is the rank of the
        // tensor type.
        std::vector<int64_t> range(shapedType.getRank());
        std::iota(range.begin(), range.end(), 0);
        if (shapedType.getRank() != layout.size() ||
            !std::is_permutation(range.begin(), range.end(), layout.begin()))
          return emitOpError()
                 << "incorrect layout " << layout << " for type " << type
                 << ", layout must be a permutation of [0, "
                 << shapedType.getRank() << ")";
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
    if (getNumResults() == 1 && isa<TupleType>(getResult(0).getType()))
      resultTypes = cast<TupleType>(getResult(0).getType()).getTypes();
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
    auto alias = cast<OutputOperandAliasAttr>(attr);
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
      if (!isa<TupleType>(operandPart) ||
          i >= static_cast<int64_t>(cast<TupleType>(operandPart).size()) ||
          i < 0)
        return emitOpError()
               << "operand_tuple_indices in the output_operand_alias "
                  "attribute out of bounds";
      operandPart = cast<TupleType>(operandPart).getType(i);
    }
    Type outputPart = getNumResults() > 1
                          ? TupleType::get(getContext(), getResultTypes())
                          : getResult(0).getType();
    for (auto i : outputTupleIndices) {
      if (!isa<TupleType>(outputPart) ||
          i >= static_cast<int64_t>(cast<TupleType>(outputPart).size()) ||
          i < 0)
        return emitOpError()
               << "output_tuple_indices in the output_operand_alias "
                  "attribute out of bounds";
      outputPart = cast<TupleType>(outputPart).getType(i);
    }
    if (operandPart != outputPart)
      return emitOpError()
             << "shapes mismatch in the output_operand_alias attribute: "
             << "operand part has type " << operandPart
             << " and output part has type " << outputPart;
  }
  if (auto backendConfig = getBackendConfig()) {
    if (getApiVersion() == CustomCallApiVersion::API_VERSION_TYPED_FFI) {
      if (!isa<mlir::DictionaryAttr>(*backendConfig))
        return emitOpError() << "backend_config for api_version "
                             << stringifyCustomCallApiVersion(getApiVersion())
                             << " must be a dictionary attribute.";
    } else {
      if (!isa<mlir::StringAttr>(*backendConfig))
        return emitOpError() << "backend_config for api_version "
                             << stringifyCustomCallApiVersion(getApiVersion())
                             << " must be a string attribute.";
    }
  }

  return success();
}

void CustomCallOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>&
        effects) {
  // CustomCall has "all possible effects" unless the has_side_effect is present
  // and set to false.
  auto hasSideEffect = getHasSideEffectAttr();
  if (hasSideEffect && !hasSideEffect.getValue()) return;
  effects.emplace_back(MemoryEffects::Allocate::get());
  effects.emplace_back(MemoryEffects::Free::get());
  effects.emplace_back(MemoryEffects::Write::get());
  effects.emplace_back(MemoryEffects::Read::get());
}

mlir::Attribute CustomCallOp::getBackendConfigOrDefault() {
  auto backendConfig = getBackendConfig();
  if (backendConfig.has_value()) return backendConfig.value();

  if (getApiVersion() ==
      mlir::stablehlo::CustomCallApiVersion::API_VERSION_TYPED_FFI)
    return DictionaryAttr::get(getContext());

  return StringAttr::get(getContext(), "");
}

// Returns if the backend config is unset, or if empty dict / string attribute.
bool CustomCallOp::hasEmptyBackendConfig() {
  if (!getBackendConfig().has_value()) return true;
  Attribute backendConfig = getBackendConfigOrDefault();
  if (auto strAttr = dyn_cast<StringAttr>(backendConfig)) {
    return strAttr.empty();
  }
  return cast<DictionaryAttr>(backendConfig).empty();
}

//===----------------------------------------------------------------------===//
// CholeskyOp
//===----------------------------------------------------------------------===//

LogicalResult CholeskyOp::inferReturnTypeComponents(
    MLIRContext*, std::optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  CholeskyOp::Adaptor adaptor(operands, attributes, properties, regions);
  return hlo::inferCholeskyOp(location, adaptor.getA(), inferredReturnShapes);
}

//===----------------------------------------------------------------------===//
// DotOp
//===----------------------------------------------------------------------===//

LogicalResult DotOp::verify() {
  return hlo::verifyDotOp(getLoc(), getLhs().getType(), getRhs().getType(),
                          getPrecisionConfig(), getResult());
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
    p << stringifyPrecision(cast<PrecisionAttr>(attr).getValue());
  });
  p << ']';
}

void printDotAlgorithm(OpAsmPrinter& p, Operation*,
                       DotAlgorithmAttr algorithm) {
  // Precision config is an optional attribute, passes null if not specified.
  if (!algorithm) return;
  p << ", algorithm = ";
  p.printStrippedAttrOrType(algorithm);
}

ParseResult parsePrecisionConfigImpl(OpAsmParser& parser,
                                     mlir::ArrayAttr& precision) {
  if (failed(parser.parseKeyword("precision")) || failed(parser.parseEqual()))
    return failure();
  SmallVector<Attribute> attrs;
  if (failed(parser.parseCommaSeparatedList(
          AsmParser::Delimiter::Square, [&]() -> ParseResult {
            attrs.push_back(PrecisionAttr::parse(parser, {}));
            return success(/*isSuccess=*/static_cast<bool>(attrs.back()));
          })))
    return failure();

  precision = mlir::ArrayAttr::get(parser.getContext(), attrs);
  return success();
}

void printPrecisionConfigAndAlgorithm(OpAsmPrinter& p, Operation* op,
                                      ::mlir::ArrayAttr precision,
                                      DotAlgorithmAttr algorithm) {
  printPrecisionConfig(p, op, precision);
  printDotAlgorithm(p, op, algorithm);
}

ParseResult parsePrecisionConfigAndAlgorithm(OpAsmParser& parser,
                                             mlir::ArrayAttr& precision,
                                             DotAlgorithmAttr& algorithm) {
  // OptPrecisionAndAlgorithm ::= `,` PrecisionAndAlgorithm || empty
  // PrecisionAndAlgorithm ::= Precision OptAlgorithm || Algorithm
  // Precision ::= `precision` `=` PrecisionConfigAttr
  // OptAlgorithm ::= `,` Algorithm || empty
  // Algorithm ::= `algorithm` `=` DotAlgorithmAttr

  // OptPrecisionAndAlgorithm
  if (failed(parser.parseOptionalComma())) return success();

  auto parseAlgorithm = [](OpAsmParser& parser,
                           DotAlgorithmAttr& algorithm) -> ParseResult {
    Type none;
    auto result = DotAlgorithmAttr::parse(parser, none);
    if (!result) return failure();
    algorithm = cast<DotAlgorithmAttr>(result);
    return success();
  };

  // PrecisionAndAlgorithm -> Algorithm
  if (succeeded(parser.parseOptionalKeyword("algorithm"))) {
    if (failed(parser.parseEqual()) ||
        failed(parseAlgorithm(parser, algorithm)))
      return failure();
    return success();
  }

  // PrecisionAndAlgorithm -> Precision OptAlgorithm
  if (failed(parsePrecisionConfigImpl(parser, precision))) return failure();

  // OptAlgorithm
  if (failed(parser.parseOptionalComma())) return success();

  // Algorithm
  if (failed(parser.parseKeyword("algorithm")) || failed(parser.parseEqual()) ||
      failed(parseAlgorithm(parser, algorithm)))
    return failure();
  return success();
}

ParseResult parsePrecisionConfig(OpAsmParser& parser,
                                 mlir::ArrayAttr& precision) {
  // OptPrecisionConfig ::= `,` Precision || empty
  if (failed(parser.parseOptionalComma())) return success();
  return parsePrecisionConfigImpl(parser, precision);
}

//===----------------------------------------------------------------------===//
// DotGeneralOp
//===----------------------------------------------------------------------===//

LogicalResult DotGeneralOp::verify() {
  bool isDefaultPrecisionConfig =
      !getPrecisionConfig().has_value() ||
      llvm::all_of(getPrecisionConfig().value(), [](Attribute attr) {
        return cast<PrecisionAttr>(attr).getValue() == Precision::DEFAULT;
      });
  bool hasAlgorithmSpecified = getAlgorithm().has_value();
  if (hasAlgorithmSpecified) {
    DotAlgorithmAttr attr = getAlgorithm().value();
    if (failed(DotAlgorithmAttr::verify(
            [&] { return this->emitError(); }, attr.getLhsPrecisionType(),
            attr.getRhsPrecisionType(), attr.getAccumulationType(),
            attr.getLhsComponentCount(), attr.getRhsComponentCount(),
            attr.getNumPrimitiveOperations(),
            attr.getAllowImpreciseAccumulation())))
      return failure();
  }

  return hlo::verifyDotGeneralOp(
      getLoc(), getLhs(), getRhs(),
      getDotDimensionNumbersAttr().getLhsBatchingDimensions(),
      getDotDimensionNumbersAttr().getRhsBatchingDimensions(),
      getDotDimensionNumbersAttr().getLhsContractingDimensions(),
      getDotDimensionNumbersAttr().getRhsContractingDimensions(),
      getPrecisionConfig(), isDefaultPrecisionConfig, hasAlgorithmSpecified,
      getResult());
}

LogicalResult DotGeneralOp::reifyReturnTypeShapes(
    OpBuilder& builder, ValueRange operands,
    SmallVectorImpl<Value>& reifiedReturnShapes) {
  auto lhsType = getLhs().getType();
  auto rhsType = getRhs().getType();

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

mlir::Speculation::Speculatability DotGeneralOp::getSpeculatability() {
  // Batching and contracting dims must be static, otherwise they could disagree
  // at runtime.
  // Other dims follow SpeculatableIfStaticDimInOutputIsStaticInInput.

  auto lhsType = getLhs().getType();
  auto rhsType = getRhs().getType();

  auto dimensionsAttr = getDotDimensionNumbersAttr();
  auto lhsBatchingDimensions = dimensionsAttr.getLhsBatchingDimensions();
  auto lhsContractingDimensions = dimensionsAttr.getLhsContractingDimensions();
  auto rhsBatchingDimensions = dimensionsAttr.getRhsBatchingDimensions();
  auto rhsContractingDimensions = dimensionsAttr.getRhsContractingDimensions();

  auto lhsSpecialDimensions = llvm::concat<const int64_t>(
      lhsBatchingDimensions, lhsContractingDimensions);
  auto rhsSpecialDimensions = llvm::concat<const int64_t>(
      rhsBatchingDimensions, rhsContractingDimensions);

  for (auto i : lhsSpecialDimensions)
    if (lhsType.isDynamicDim(i)) return mlir::Speculation::NotSpeculatable;
  for (auto i : rhsSpecialDimensions)
    if (rhsType.isDynamicDim(i)) return mlir::Speculation::NotSpeculatable;

  auto resultType = getType();
  int64_t resultIndex = lhsBatchingDimensions.size();
  for (int64_t i = 0; i < lhsType.getRank(); i++) {
    if (llvm::is_contained(lhsSpecialDimensions, i)) continue;
    if (!resultType.isDynamicDim(resultIndex) && lhsType.isDynamicDim(i))
      return mlir::Speculation::NotSpeculatable;
    resultIndex++;
  }
  for (int64_t i = 0; i < rhsType.getRank(); i++) {
    if (llvm::is_contained(rhsSpecialDimensions, i)) continue;
    if (!resultType.isDynamicDim(resultIndex) && rhsType.isDynamicDim(i))
      return mlir::Speculation::NotSpeculatable;
    resultIndex++;
  }

  return mlir::Speculation::Speculatable;
}

DotDimensionNumbersAttr getDefaultDotDimensionNumbers(mlir::Value lhs) {
  return DotDimensionNumbersAttr::get(
      lhs.getContext(),
      /*lhsBatchingDimensions=*/{},
      /*rhsBatchingDimensions=*/{},
      /*lhsContractingDimensions=*/
      {cast<ShapedType>(lhs.getType()).getRank() - 1},
      /*rhsContractingDimensions=*/{0});
}

bool DotGeneralOp::isSimpleDot() {
  auto lhsRank = cast<ShapedType>(getLhs().getType()).getRank();
  auto rhsRank = cast<ShapedType>(getRhs().getType()).getRank();
  return lhsRank <= 2 && rhsRank <= 2 &&
         getDotDimensionNumbersAttr() ==
             getDefaultDotDimensionNumbers(getLhs()) &&
         !getAlgorithm().has_value();
}

LogicalResult DotAlgorithmAttr::verify(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    Type lhsPrecisionType, Type rhsPrecisionType, Type accumulationType,
    int64_t lhsComponentCount, int64_t rhsComponentCount,
    int64_t numPrimitiveOperations, bool allowImpreciseAccumulation) {
  return hlo::verifyDotAlgorithmAttr(
      emitError, lhsPrecisionType, rhsPrecisionType, accumulationType,
      lhsComponentCount, rhsComponentCount, numPrimitiveOperations,
      allowImpreciseAccumulation);
}

//===----------------------------------------------------------------------===//
// FftOp
//===----------------------------------------------------------------------===//

LogicalResult FftOp::inferReturnTypeComponents(
    MLIRContext*, std::optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  FftOp::Adaptor adaptor(operands, attributes, properties, regions);
  return hlo::inferFftOp(location, adaptor.getOperand(),
                         adaptor.getFftType() == FftType::RFFT,
                         adaptor.getFftType() == FftType::IRFFT,
                         adaptor.getFftLength(), inferredReturnShapes);
}

mlir::Speculation::Speculatability FftOp::getSpeculatability() {
  // This is the same logic as SpeculatableIfStaticDimInOutputIsStaticInInput,
  // except that for RFFT and IRFFT the last `fft_length.size()` dimensions in
  // the operand need to be static.
  auto inputType = getOperand().getType();
  auto resultType = getType();
  size_t minStaticDim = inputType.getRank();
  if (getFftType() == FftType::RFFT || getFftType() == FftType::IRFFT)
    minStaticDim = minStaticDim - getFftLength().size();
  for (size_t i : llvm::seq(inputType.getRank())) {
    if (i >= minStaticDim && inputType.isDynamicDim(i))
      return mlir::Speculation::NotSpeculatable;
    if (!resultType.isDynamicDim(i) && inputType.isDynamicDim(i))
      return mlir::Speculation::NotSpeculatable;
  }
  return mlir::Speculation::Speculatable;
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
  for (int64_t val : gather->getSliceSizes())
    sliceSizes.push_back(builder.create<arith::ConstantIndexOp>(loc, val));
}

void getSliceSizeValues(DynamicGatherOp* /*dGather*/, OpBuilder& builder,
                        Location loc, ValueRange operands,
                        SmallVectorImpl<Value>& sliceSizeValues) {
  DynamicGatherOp::Adaptor adaptor(operands);
  Value sliceSizes = adaptor.getSliceSizes();
  auto sliceSizesTy = cast<ShapedType>(sliceSizes.getType());
  for (int64_t i = 0; i < sliceSizesTy.getDimSize(0); ++i) {
    Value idx = builder.create<arith::ConstantIndexOp>(loc, i);
    sliceSizeValues.push_back(
        builder.create<tensor::ExtractOp>(loc, sliceSizes, idx));
  }
}

template <typename Op>
LogicalResult reifyGatherShape(Op* op, OpBuilder& builder, ValueRange operands,
                               SmallVectorImpl<Value>& reifiedReturnShapes) {
  auto resultTy = cast<RankedTensorType>(op->getResult().getType());

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
                           op->getDimensionNumbers().getOperandBatchingDims(),
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
    ValueShapeRange operands, DictionaryAttr attributes,
    OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  GatherOp::Adaptor adaptor(operands, attributes, properties, regions);
  return hlo::inferGatherOp(
      location, adaptor.getOperand(), adaptor.getStartIndices(),
      adaptor.getDimensionNumbers().getOffsetDims(),
      adaptor.getDimensionNumbers().getCollapsedSliceDims(),
      adaptor.getDimensionNumbers().getOperandBatchingDims(),
      adaptor.getDimensionNumbers().getStartIndicesBatchingDims(),
      adaptor.getDimensionNumbers().getStartIndexMap(),
      adaptor.getDimensionNumbers().getIndexVectorDim(),
      adaptor.getSliceSizes(), inferredReturnShapes);
}

mlir::Speculation::Speculatability GatherOp::getSpeculatability() {
  // When indices_are_sorted is true, if the start_indices are not sorted, the
  // behavior is undefined.
  // A possible improvement would be to check if the start_indices are constant
  // and if they are sorted, do not return NotSpeculatable. However, such a
  // check could be somewhat costly and has unclear ROI.
  if (getIndicesAreSorted()) return mlir::Speculation::NotSpeculatable;
  return llvm::all_of(
             this->getOperation()->getOperandTypes(),
             [](Type t) { return cast<RankedTensorType>(t).hasStaticShape(); })
             ? mlir::Speculation::Speculatable
             : mlir::Speculation::NotSpeculatable;
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
    ValueShapeRange operands, DictionaryAttr attributes,
    OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  DynamicGatherOp::Adaptor adaptor(operands, attributes, properties, regions);
  return hlo::inferDynamicGatherOp(
      location, adaptor.getOperand(), adaptor.getStartIndices(),
      adaptor.getSliceSizes(), adaptor.getDimensionNumbers().getOffsetDims(),
      adaptor.getDimensionNumbers().getCollapsedSliceDims(),
      adaptor.getDimensionNumbers().getOperandBatchingDims(),
      adaptor.getDimensionNumbers().getStartIndicesBatchingDims(),
      adaptor.getDimensionNumbers().getStartIndexMap(),
      adaptor.getDimensionNumbers().getIndexVectorDim(), inferredReturnShapes);
}

//===----------------------------------------------------------------------===//
// GetDimensionSizeOp
//===----------------------------------------------------------------------===//

LogicalResult GetDimensionSizeOp::inferReturnTypeComponents(
    MLIRContext*, std::optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  GetDimensionSizeOp::Adaptor adaptor(operands, attributes, properties,
                                      regions);
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
      builder.getContext(), cast<ShapedType>(shapeOp.getType()).getDimSize(0));
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

LogicalResult DynamicIotaOp::verify() {
  return hlo::verifyDynamicIotaOp(getLoc(), getOutputShape(),
                                  getIotaDimension(), getResult());
}

mlir::Speculation::Speculatability DynamicIotaOp::getSpeculatability() {
  // If the output shape operand is constant, each of its dimensions is static.
  // For each dimension in the result type's shape:
  // 1. If it is static, the verifier has already checked that it matches the
  //    corresponding dimension in the output shape operand.
  // 2. Otherwise, it is dynamic, so there cannot be a mismatch.
  // (In fact, the result type's shape can be inferred from the operand.)
  if (matchPattern(getOperand(), m_Constant()))
    return mlir::Speculation::Speculatable;

  // The result type's shape is fully dynamic, so there cannot be a mismatch
  // with the output shape operand at runtime (the type has no expectations).
  if (llvm::all_of(llvm::seq(getType().getRank()),
                   [this](int64_t i) { return getType().isDynamicDim(i); }))
    return mlir::Speculation::Speculatable;

  // The output shape operand's value is unknown and at least one of the result
  // type's dimensions is static, so the dimensions could disagree at runtime.
  return mlir::Speculation::NotSpeculatable;
}

//===----------------------------------------------------------------------===//
// DynamicUpdateSliceOp
//===----------------------------------------------------------------------===//

LogicalResult DynamicUpdateSliceOp::inferReturnTypeComponents(
    MLIRContext*, std::optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  DynamicUpdateSliceOp::Adaptor adaptor(operands, attributes, properties,
                                        regions);
  return hlo::inferDynamicUpdateSliceOp(
      location, adaptor.getOperand(), adaptor.getUpdate(),
      adaptor.getStartIndices(), inferredReturnShapes);
}

//===----------------------------------------------------------------------===//
// AbsOp
//===----------------------------------------------------------------------===//

LogicalResult AbsOp::inferReturnTypes(
    MLIRContext*, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  AbsOp::Adaptor adaptor(operands, attributes, properties, regions);
  return hlo::inferAbsOp(location, adaptor.getOperand(), inferredReturnTypes);
}

//===----------------------------------------------------------------------===//
// CollectiveBroadcastOp
//===----------------------------------------------------------------------===//

void CollectiveBroadcastOp::build(OpBuilder& odsBuilder,
                                  OperationState& odsState, Type resultType,
                                  Value operand,
                                  DenseIntElementsAttr replica_groups) {
  CollectiveBroadcastOp::build(odsBuilder, odsState, resultType, operand,
                               replica_groups, /*channel_handle=*/nullptr);
}

LogicalResult CollectiveBroadcastOp::inferReturnTypes(
    MLIRContext* /*context*/, std::optional<Location> location,
    ValueRange operands, DictionaryAttr attributes, OpaqueProperties properties,
    RegionRange regions, SmallVectorImpl<Type>& inferredReturnTypes) {
  CollectiveBroadcastOp::Adaptor adaptor(operands, attributes, properties,
                                         regions);
  return hlo::inferCollectiveBroadcastOp(location, adaptor.getOperands(),
                                         inferredReturnTypes);
}

LogicalResult CollectiveBroadcastOp::inferReturnTypeComponents(
    MLIRContext* context, std::optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes,
    OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  SmallVector<Type> inferredReturnTypes;
  CollectiveBroadcastOp::Adaptor adaptor(operands, attributes, properties,
                                         regions);
  if (failed(hlo::inferCollectiveBroadcastOp(location, adaptor.getOperands(),
                                             inferredReturnTypes)))
    return failure();
  if (inferredReturnTypes.size() != 1) return failure();
  auto inferredReturnType = dyn_cast<ShapedType>(inferredReturnTypes[0]);
  if (!inferredReturnType) return failure();
  inferredReturnShapes.push_back(inferredReturnType);
  return success();
}

LogicalResult CollectiveBroadcastOp::verify() {
  return hlo::verifyCollectiveBroadcastOp(getLoc(), getReplicaGroups());
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

LogicalResult CollectivePermuteOp::inferReturnTypes(
    MLIRContext* /*context*/, std::optional<Location> location,
    ValueRange operands, DictionaryAttr attributes, OpaqueProperties properties,
    RegionRange regions, SmallVectorImpl<Type>& inferredReturnTypes) {
  CollectivePermuteOp::Adaptor adaptor(operands, attributes, properties,
                                       regions);
  return hlo::inferCollectivePermuteOp(location, adaptor.getOperands(),
                                       inferredReturnTypes);
}

LogicalResult CollectivePermuteOp::inferReturnTypeComponents(
    MLIRContext* context, std::optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes,
    OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  SmallVector<Type> inferredReturnTypes;
  CollectivePermuteOp::Adaptor adaptor(operands, attributes, properties,
                                       regions);
  if (failed(hlo::inferCollectivePermuteOp(location, adaptor.getOperands(),
                                           inferredReturnTypes)))
    return failure();
  if (inferredReturnTypes.size() != 1) return failure();
  auto inferredReturnType = dyn_cast<ShapedType>(inferredReturnTypes[0]);
  if (!inferredReturnType) return failure();
  inferredReturnShapes.push_back(inferredReturnType);
  return success();
}

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

mlir::Speculation::Speculatability ConvolutionOp::getSpeculatability() {
  auto inputType = getLhs().getType();
  auto kernelType = getRhs().getType();
  auto resultType = getType();

  auto dimNumbers = getDimensionNumbers();
  auto inputBatchDim = dimNumbers.getInputBatchDimension();
  auto inputFeatureDim = dimNumbers.getInputFeatureDimension();
  auto inputSpatialDims = dimNumbers.getInputSpatialDimensions();
  auto kernelInputFeatureDim = dimNumbers.getKernelInputFeatureDimension();
  auto kernelOutputFeatureDim = dimNumbers.getKernelOutputFeatureDimension();
  auto kernelSpatialDims = dimNumbers.getKernelSpatialDimensions();
  auto outputBatchDim = dimNumbers.getOutputBatchDimension();
  auto outputFeatureDim = dimNumbers.getOutputFeatureDimension();
  auto outputSpatialDims = dimNumbers.getOutputSpatialDimensions();

  auto batchGroupCount = getBatchGroupCount();
  auto featureGroupCount = getFeatureGroupCount();

  // input_feature_dimension and kernel_input_feature_dimension must be static
  // (C14).
  if (inputType.isDynamicDim(inputFeatureDim) ||
      kernelType.isDynamicDim(kernelInputFeatureDim))
    return mlir::Speculation::NotSpeculatable;

  // input_batch_dimension must be static if batch_group_count > 1 (C10) or if
  // output_batch_dimension is static (C25).
  if (inputType.isDynamicDim(inputBatchDim) &&
      (batchGroupCount > 1 || !resultType.isDynamicDim(outputBatchDim)))
    return mlir::Speculation::NotSpeculatable;

  // kernel_output_feature_dimension must be static if batch_group_count > 1
  // (C15) or feature_group_count > 1 (C16) or if output_feature_dimension is
  // static (C25).
  if (kernelType.isDynamicDim(kernelOutputFeatureDim) &&
      (batchGroupCount > 1 || featureGroupCount > 1 ||
       !resultType.isDynamicDim(outputFeatureDim)))
    return mlir::Speculation::NotSpeculatable;

  // If a spatial dimension is static in the output, it must be static in the
  // inputs (C25).
  for (auto [inputDim, kernelDim, resultDim] :
       llvm::zip(inputSpatialDims, kernelSpatialDims, outputSpatialDims)) {
    if (!resultType.isDynamicDim(resultDim) &&
        (inputType.isDynamicDim(inputDim) ||
         kernelType.isDynamicDim(kernelDim)))
      return mlir::Speculation::NotSpeculatable;
  }

  return mlir::Speculation::Speculatable;
}

//===----------------------------------------------------------------------===//
// ConvertOp
//===----------------------------------------------------------------------===//

void ConvertOp::build(OpBuilder& builder, OperationState& result, Value operand,
                      Type resultElementTy) {
  auto rankedTy = cast<RankedTensorType>(operand.getType());
  auto resultTy = RankedTensorType::get(rankedTy.getShape(), resultElementTy);
  build(builder, result, resultTy, operand);
}

//===----------------------------------------------------------------------===//
// AllToAllOp
//===----------------------------------------------------------------------===//

LogicalResult AllToAllOp::inferReturnTypeComponents(
    MLIRContext*, std::optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  AllToAllOp::Adaptor adaptor(operands, attributes, properties, regions);
  return hlo::inferAllToAllOp(
      location, adaptor.getOperands(), adaptor.getSplitDimension(),
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

mlir::Speculation::Speculatability AllToAllOp::getSpeculatability() {
  for (auto [operand, result] : llvm::zip(getOperands(), getResults())) {
    auto inputType = cast<RankedTensorType>(operand.getType());
    auto resultType = cast<RankedTensorType>(result.getType());
    auto splitDim = getSplitDimension();
    auto concatDim = getConcatDimension();
    // The actual size of the `splitDim` and `concatDim` depends on the number
    // of processes, which is only known at runtime. If it is dynamic, there is
    // no expectation, so there cannot be a mismatch. If it is static, the
    // actual number may differ at runtime, leading to UB. See all_to_all_c9 in
    // the spec.
    if (!resultType.isDynamicDim(splitDim) ||
        !resultType.isDynamicDim(concatDim))
      return mlir::Speculation::NotSpeculatable;
    for (size_t i : llvm::seq(resultType.getRank())) {
      if (i == splitDim || i == concatDim) continue;
      if (!resultType.isDynamicDim(i) && inputType.isDynamicDim(i))
        return mlir::Speculation::NotSpeculatable;
    }
  }
  return mlir::Speculation::Speculatable;
}

//===----------------------------------------------------------------------===//
// AllGatherOp
//===----------------------------------------------------------------------===//

LogicalResult AllGatherOp::verify() {
  int64_t channelId = 0;
  if (auto channelHandleAttr = getChannelHandleAttr())
    channelId = channelHandleAttr.getHandle();

  return hlo::verifyAllGatherOp(getLoc(), getOperands(), getAllGatherDim(),
                                getReplicaGroups(), channelId,
                                getUseGlobalDeviceIds(), getResults());
}

mlir::Speculation::Speculatability AllGatherOp::getSpeculatability() {
  for (auto [operand, result] : llvm::zip(getOperands(), getResults())) {
    auto inputType = cast<RankedTensorType>(operand.getType());
    auto resultType = cast<RankedTensorType>(result.getType());
    auto allGatherDim = getAllGatherDim();
    // The actual size of the `allGatherDim` depends on the number of processes,
    // which is only known at runtime. If it is dynamic, there is no
    // expectation, so there cannot be a mismatch. If it is static, the actual
    // number may differ at runtime, leading to UB. See all_gather_c6 in the
    // spec.
    if (!resultType.isDynamicDim(allGatherDim))
      return mlir::Speculation::NotSpeculatable;
    for (size_t i : llvm::seq(resultType.getRank())) {
      if (i != allGatherDim && !resultType.isDynamicDim(i) &&
          inputType.isDynamicDim(i))
        return mlir::Speculation::NotSpeculatable;
    }
  }
  return mlir::Speculation::Speculatable;
}

//===----------------------------------------------------------------------===//
// AllReduceOp
//===----------------------------------------------------------------------===//

void AllReduceOp::build(OpBuilder& odsBuilder, OperationState& odsState,
                        Type resultType, Value operand,
                        DenseIntElementsAttr replicaGroups,
                        ChannelHandleAttr channelHandle,
                        bool useGlobalDeviceIds) {
  build(odsBuilder, odsState, resultType, ValueRange(operand), replicaGroups,
        channelHandle, useGlobalDeviceIds);
}

LogicalResult AllReduceOp::verify() {
  int64_t channelId = 0;
  if (auto channelHandleAttr = getChannelHandleAttr())
    channelId = channelHandleAttr.getHandle();

  return hlo::verifyAllReduceOp(getLoc(), getOperands(), getReplicaGroups(),
                                channelId, getUseGlobalDeviceIds(),
                                getComputation());
}

LogicalResult AllReduceOp::inferReturnTypeComponents(
    MLIRContext*, std::optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  AllReduceOp::Adaptor adaptor(operands, attributes, properties, regions);
  return hlo::inferAllReduceOp(location, adaptor.getOperands(),
                               adaptor.getComputation(), inferredReturnShapes);
}

//===----------------------------------------------------------------------===//
// BatchNormGradOp
//===----------------------------------------------------------------------===//

LogicalResult BatchNormGradOp::inferReturnTypeComponents(
    MLIRContext* context, std::optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes,
    OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  BatchNormGradOp::Adaptor adaptor(operands, attributes, properties, regions);
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
    ValueShapeRange operands, DictionaryAttr attributes,
    OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  BatchNormTrainingOp::Adaptor adaptor(operands, attributes, properties,
                                       regions);
  return hlo::inferBatchNormTrainingOp(
      location, adaptor.getOperand(), adaptor.getScale(), adaptor.getOffset(),
      adaptor.getFeatureIndex(), inferredReturnShapes);
}

//===----------------------------------------------------------------------===//
// BatchNormInferenceOp
//===----------------------------------------------------------------------===//

LogicalResult BatchNormInferenceOp::inferReturnTypeComponents(
    MLIRContext* context, std::optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes,
    OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  BatchNormInferenceOp::Adaptor adaptor(operands, attributes, properties,
                                        regions);
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
  auto operandType = cast<RankedTensorType>(operands[0].getType());
  auto resultType = getType();

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

mlir::Speculation::Speculatability BitcastConvertOp::getSpeculatability() {
  // The logic is the same as for the
  // SpeculatableIfStaticdimInOutputIsStaticInInput trait, except we don't need
  // to check any "extra" dimension that may result from the difference in bit
  // width of the input and result. Indeed, the extra dimension can be deduced
  // from the bit widths.
  auto inputType = getOperand().getType();
  auto resultType = getType();
  auto rank = std::min(inputType.getRank(), resultType.getRank());
  for (size_t i : llvm::seq(rank)) {
    if (!resultType.isDynamicDim(i) && inputType.isDynamicDim(i))
      return mlir::Speculation::NotSpeculatable;
  }
  return mlir::Speculation::Speculatable;
}

//===----------------------------------------------------------------------===//
// BroadcastOp
//===----------------------------------------------------------------------===//

LogicalResult BroadcastOp::inferReturnTypeComponents(
    MLIRContext*, std::optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  BroadcastOp::Adaptor adaptor(operands, attributes, properties, regions);
  return hlo::inferBroadcastOp(location, adaptor.getOperand(),
                               adaptor.getBroadcastSizes(),
                               inferredReturnShapes);
}

LogicalResult BroadcastOp::reifyReturnTypeShapes(
    OpBuilder& builder, ValueRange operands,
    SmallVectorImpl<Value>& reifiedReturnShapes) {
  BroadcastOp::Adaptor adaptor(operands);
  Value operand = adaptor.getOperand();

  auto operandType = cast<RankedTensorType>(operand.getType());

  Location loc = getLoc();
  SmallVector<Value, 4> shapeValues;

  // Collect the broadcast sizes.
  for (const auto& size : getBroadcastSizes())
    shapeValues.push_back(builder.create<arith::ConstantIndexOp>(loc, size));

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

// Creates BroadcastInDimOp.broadcast_dimensions from BroadcastOp using the
// number of broadcast_sizes and the rank of the operand.
SmallVector<int64_t> getBroadcastDimensionsFromBroadcast(
    int64_t broadcastSizesSize, int64_t operandRank) {
  return llvm::to_vector(
      llvm::seq(broadcastSizesSize, broadcastSizesSize + operandRank));
}

DenseI64ArrayAttr getBroadcastDimensionsFromBroadcastSizes(
    RankedTensorType resultType, DenseI64ArrayAttr broadcastSizes) {
  int64_t broadcastSizesSize = broadcastSizes.size();
  int64_t operandRank = resultType.getRank() - broadcastSizesSize;
  return DenseI64ArrayAttr::get(
      resultType.getContext(),
      getBroadcastDimensionsFromBroadcast(broadcastSizesSize, operandRank));
}

namespace {

// Check that broadcast dimensions are suitable for isSimpleBroadcast():
// extending rank is OK for them, but for dims where rank is not extended the
// dim sizes must match.
//
// Two dimensions are compatible when:
// - they are equal, or
// - one of them is 1.
// and here we reject the case when only one of them is 1.
//
// Examples of compatible dimensions for simple broadcast:
// - tensor<3xf32> -> tensor<1x2x3xf32>
// - tensor<1x1xf32> -> tensor<3x1x1xf32>
// - tensor<5x7xf32> -> tensor<1x3x5x7xf32>
// Examples of non-compatible dimensions:
// - tensor<3xf32> -> tensor<3x3xf32>
// - tensor<3xf32> -> tensor<1x3x3xf32>
// - tensor<1x1xf32> -> tensor<3x2x2xf32>
// - tensor<3x1x1xf32> -> tensor<1x3x5x7xf32>
// - tensor<1x5x7xf32> -> tensor<1x3x5x7xf32>
bool haveSimpleCompatibleDimensions(RankedTensorType operand,
                                    RankedTensorType result) {
  auto operandTy = cast<ShapedType>(operand);
  auto resultTy = cast<ShapedType>(result);
  ArrayRef<int64_t> operandShape = operandTy.getShape();
  ArrayRef<int64_t> resultShape = resultTy.getShape();
  bool isCompatible = true;
  for (auto [operandDim, resultDim] : llvm::zip(operandShape, resultShape))
    isCompatible &= operandDim == resultDim;
  return isCompatible;
}
}  // namespace

bool BroadcastInDimOp::isSimpleBroadcast() {
  auto operandTy = getOperand().getType();
  auto resultTy = getType();
  auto operandRank = operandTy.getRank();
  auto broadcastSizesSize = resultTy.getRank() - operandRank;
  bool haveCompatibleDimensions =
      haveSimpleCompatibleDimensions(operandTy, resultTy);
  return haveCompatibleDimensions &&
         llvm::to_vector(getBroadcastDimensions()) ==
             getBroadcastDimensionsFromBroadcast(broadcastSizesSize,
                                                 operandRank);
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

mlir::Speculation::Speculatability
DynamicBroadcastInDimOp::getSpeculatability() {
  auto operandType = getOperand().getType();

  // If input is dynamic, the broadcasting rules might be violated at runtime,
  // so not speculatable.
  if (!operandType.hasStaticShape()) return mlir::Speculation::NotSpeculatable;

  // If input is broadcastable (all 1's) and result is fully dynamic,
  // speculatable.
  auto resultDynamic =
      llvm::all_of(llvm::seq(getType().getRank()),
                   [this](int64_t i) { return getType().isDynamicDim(i); });
  if (operandType.getNumElements() == 1 && resultDynamic)
    return mlir::Speculation::Speculatable;

  // If shape is known, speculatable.
  if (matchPattern(getOutputDimensions(), m_Constant()))
    return mlir::Speculation::Speculatable;

  return mlir::Speculation::NotSpeculatable;
}

//===----------------------------------------------------------------------===//
// DynamicBroadcastInDimOp
//===----------------------------------------------------------------------===//

LogicalResult DynamicConvOp::verify() {
  return hlo::verifyDynamicConvOp(
      getLoc(), getLhs().getType(), getRhs().getType(), getPadding(),
      getWindowStrides(), getLhsDilation(), getRhsDilation(),
      getWindowReversal(), getDimensionNumbers().getInputBatchDimension(),
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
// ClampOp
//===----------------------------------------------------------------------===//

LogicalResult ClampOp::inferReturnTypeComponents(
    MLIRContext*, std::optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  ClampOp::Adaptor adaptor(operands, attributes, properties, regions);
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
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  ComplexOp::Adaptor adaptor(operands, attributes, properties, regions);
  return hlo::inferComplexOp(location, adaptor.getLhs(), inferredReturnTypes);
}

//===----------------------------------------------------------------------===//
// ImagOp
//===----------------------------------------------------------------------===//

LogicalResult ImagOp::inferReturnTypes(
    MLIRContext*, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  ImagOp::Adaptor adaptor(operands, attributes, properties, regions);
  return hlo::inferImagOp(location, adaptor.getOperand(), inferredReturnTypes);
}

//===----------------------------------------------------------------------===//
// IsFiniteOp
//===----------------------------------------------------------------------===//

LogicalResult IsFiniteOp::inferReturnTypes(
    MLIRContext* ctx, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  IsFiniteOp::Adaptor adaptor(operands, attributes, properties, regions);
  return hlo::inferIsFiniteOp(ctx, location, adaptor.getX(),
                              inferredReturnTypes);
}

//===----------------------------------------------------------------------===//
// RealOp
//===----------------------------------------------------------------------===//

LogicalResult RealOp::inferReturnTypes(
    MLIRContext*, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  RealOp::Adaptor adaptor(operands, attributes, properties, regions);
  return hlo::inferRealOp(location, adaptor.getOperand(), inferredReturnTypes);
}

//===----------------------------------------------------------------------===//
// ConcatenateOp
//===----------------------------------------------------------------------===//

LogicalResult ConcatenateOp::inferReturnTypes(
    MLIRContext*, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  ConcatenateOp::Adaptor adaptor(operands, attributes, properties, regions);
  return hlo::inferConcatenateOp(location, adaptor.getInputs().getTypes(),
                                 adaptor.getDimension(), inferredReturnTypes);
}

LogicalResult ConcatenateOp::reifyReturnTypeShapes(
    OpBuilder& builder, ValueRange operands,
    SmallVectorImpl<Value>& reifiedReturnShapes) {
  ConcatenateOp::Adaptor adaptor(operands);
  auto inputs = adaptor.getInputs();

  Location loc = this->getLoc();
  Type shapeScalarType = builder.getIndexType();
  auto toShapeScalarType = [&](Value v) {
    return maybeCastTo(builder, loc, v, shapeScalarType);
  };

  SmallVector<SmallVector<Value, 4>, 4> allShapeValues;
  for (size_t inputId = 0; inputId < inputs.size(); ++inputId) {
    Value operand = inputs[inputId];
    auto operandType = cast<RankedTensorType>(operand.getType());

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

mlir::Speculation::Speculatability ConcatenateOp::getSpeculatability() {
  // All operand dimensions must be static, except maybe the concat dim.
  // If concat dim is dynamic, the corresponding dim in operands can be dynamic,
  // otherwise it has to be static.
  auto concatDim = getDimension();
  bool concatDimDynamic = getType().isDynamicDim(concatDim);
  for (auto t : getOperandTypes()) {
    auto rankedT = cast<RankedTensorType>(t);
    for (uint64_t i : llvm::seq(rankedT.getRank())) {
      if (i == concatDim && concatDimDynamic) continue;
      if (rankedT.isDynamicDim(i)) return mlir::Speculation::NotSpeculatable;
    }
  }
  return mlir::Speculation::Speculatable;
}

//===----------------------------------------------------------------------===//
// DynamicReshapeOp
//===----------------------------------------------------------------------===//

LogicalResult DynamicReshapeOp::verify() {
  return hlo::verifyDynamicReshapeOp(getLoc(), getOperand(), getOutputShape(),
                                     getResult());
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
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  DynamicSliceOp::Adaptor adaptor(operands, attributes, properties, regions);
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

  auto operandType = cast<RankedTensorType>(operand.getType());

  Location loc = this->getLoc();
  SmallVector<Value, 4> shapeValues;
  shapeValues.reserve(operandType.getRank());
  Type shapeScalarType =
      cast<ShapedType>(startIndices.getType()).getElementType();
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

mlir::Speculation::Speculatability RealDynamicSliceOp::getSpeculatability() {
  return hlo::getShapedSpeculatability(getOperation(), /*shapeCount=*/3);
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
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  MapOp::Adaptor adaptor(operands, attributes, properties, regions);
  return hlo::inferMapOp(location, adaptor.getInputs(), adaptor.getDimensions(),
                         adaptor.getComputation(), inferredReturnShapes);
}

LogicalResult MapOp::reifyReturnTypeShapes(
    OpBuilder& builder, ValueRange operands,
    SmallVectorImpl<Value>& reifiedReturnShapes) {
  return hlo::deriveShapeFromOperand(&builder, getOperation(), operands.front(),
                                     &reifiedReturnShapes);
}

mlir::Speculation::Speculatability MapOp::getSpeculatability() {
  // If any dimension of any operand is dynamic, it could disagree with the
  // others at runtime, so the op is not speculatable. If all the operands are
  // statically shaped, whether the op is speculatable or not depends on what
  // ops are in the op's body.
  return llvm::all_of(
             this->getOperation()->getOperandTypes(),
             [](Type t) { return cast<RankedTensorType>(t).hasStaticShape(); })
             ? mlir::Speculation::RecursivelySpeculatable
             : mlir::Speculation::NotSpeculatable;
}

//===----------------------------------------------------------------------===//
// OutfeedOp
//===----------------------------------------------------------------------===//

LogicalResult OutfeedOp::inferReturnTypes(
    MLIRContext* context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  return hlo::inferOutfeedOp(getStablehloDialect(context), location,
                             inferredReturnTypes);
}

//===----------------------------------------------------------------------===//
// Send Op
//===----------------------------------------------------------------------===//

LogicalResult SendOp::inferReturnTypes(
    MLIRContext* context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  SendOp::Adaptor adaptor(operands, attributes, properties, regions);
  bool isDeviceToDevice = adaptor.getChannelHandle().getType() == 1;
  bool isDeviceToHost = adaptor.getChannelHandle().getType() == 2;
  return hlo::inferSendOp(getStablehloDialect(context), location,
                          isDeviceToDevice, isDeviceToHost,
                          adaptor.getIsHostTransfer(), inferredReturnTypes);
}

//===----------------------------------------------------------------------===//
// RecvOp
//===----------------------------------------------------------------------===//

LogicalResult RecvOp::verify() {
  bool isDeviceToDevice = getChannelHandle().getType() == 1;
  bool isHostToDevice = getChannelHandle().getType() == 3;
  return hlo::verifyRecvOp(getStablehloDialect(getContext()), getLoc(),
                           isDeviceToDevice, isHostToDevice,
                           getIsHostTransfer(), getResults());
}

//===----------------------------------------------------------------------===//
// ReduceWindowOp
//===----------------------------------------------------------------------===//

LogicalResult ReduceWindowOp::inferReturnTypeComponents(
    MLIRContext*, std::optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  ReduceWindowOp::Adaptor adaptor(operands, attributes, properties, regions);
  return hlo::inferReduceWindowOp(
      location, adaptor.getInputs(), adaptor.getInitValues(),
      adaptor.getWindowDimensions(), adaptor.getWindowStrides(),
      adaptor.getBaseDilations(), adaptor.getWindowDilations(),
      adaptor.getPadding(), adaptor.getBody(), inferredReturnShapes);
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
    ValueRange init_values, DenseI64ArrayAttr window_dimensions,
    /*optional*/ DenseI64ArrayAttr window_strides,
    /*optional*/ DenseI64ArrayAttr base_dilations,
    /*optional*/ DenseI64ArrayAttr window_dilations,
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
    auto iType = cast<ShapedType>(i.getType());
    blockArgTypes.push_back(iType.cloneWith(
        llvm::ArrayRef<int64_t>(std::nullopt), iType.getElementType()));
    locs.push_back(i.getLoc());
  }
  for (auto i : init_values) {
    auto iType = cast<ShapedType>(i.getType());
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
          odsState.getRawProperties(), odsState.regions, inferredReturnTypes)))
    odsState.addTypes(inferredReturnTypes);
  else
    llvm::report_fatal_error("Failed to infer result type(s).");
}

//===----------------------------------------------------------------------===//
// ReduceOp
//===----------------------------------------------------------------------===//

void ReduceOp::print(OpAsmPrinter& p) {
  hlo::printReduceOp(p, getOperation(), getInputs(), getDimensions(),
                     getBody());
}

ParseResult ReduceOp::parse(OpAsmParser& parser, OperationState& result) {
  auto parseDenseArray = [](OpBuilder& b, ArrayRef<int64_t> dims) -> Attribute {
    return b.getDenseI64ArrayAttr(dims);
  };
  return hlo::parseReduceOp(parser, result, parseDenseArray);
}

LogicalResult ReduceOp::inferReturnTypeComponents(
    MLIRContext*, std::optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  ReduceOp::Adaptor adaptor(operands, attributes, properties, regions);
  return hlo::inferReduceOp(location, adaptor.getInputs().getTypes(),
                            adaptor.getDimensions(), adaptor.getBody(),
                            inferredReturnShapes);
}

void ReduceOp::build(OpBuilder&, OperationState& odsState, ValueRange inputs,
                     ValueRange initValues, DenseI64ArrayAttr dimensions,
                     TypeRange elementTypes) {
  odsState.addOperands(inputs);
  odsState.addOperands(initValues);
  odsState.addAttribute(getDimensionsAttrName(odsState.name), dimensions);
  (void)odsState.addRegion();

  SmallVector<int64_t> newDimensions;
  Attribute encoding;
  ReduceOp::Adaptor adaptor(
      odsState.operands,
      odsState.attributes.getDictionary(odsState.getContext()), {},
      odsState.regions);

  auto inputArgTensorTypes =
      llvm::map_to_vector(adaptor.getInputs().getTypes(),
                          [](Type t) { return cast<ShapedType>(t); });
  auto initValueTensorTypes =
      llvm::map_to_vector(adaptor.getInitValues().getTypes(),
                          [](Type t) { return cast<ShapedType>(t); });

  if (failed(hlo::verifyReduceOpInputsAndInferShape(
          odsState.location, inputArgTensorTypes, dimensions, newDimensions,
          encoding)))
    llvm::report_fatal_error("Failed to infer result type(s).");

  SmallVector<Type> inferredReturnTypes;
  for (auto [inputTy, elementTy] :
       llvm::zip(inputArgTensorTypes, elementTypes)) {
    inferredReturnTypes.push_back(
        RankedTensorType::get(newDimensions, elementTy, encoding));
  }
  odsState.addTypes(inferredReturnTypes);
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

  auto operandType = cast<RankedTensorType>(inputs[0].getType());

  Location loc = this->getLoc();
  SmallVector<Value, 4> shapeValues;
  SmallVector<int64_t, 4> dimensions(this->getDimensions());
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
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  OptimizationBarrierOp::Adaptor adaptor(operands, attributes, properties);
  return hlo::inferOptimizationBarrierOp(location, adaptor.getOperand(),
                                         inferredReturnTypes);
}

//===----------------------------------------------------------------------===//
// ReverseOp
//===----------------------------------------------------------------------===//
LogicalResult ReverseOp::verify() {
  return hlo::verifyReverseOp(getLoc(), getOperand(), getDimensions());
}

LogicalResult ReverseOp::inferReturnTypes(
    MLIRContext* /*context*/, std::optional<Location> location,
    ValueRange operands, DictionaryAttr attributes, OpaqueProperties properties,
    RegionRange regions, SmallVectorImpl<Type>& inferredReturnTypes) {
  ReverseOp::Adaptor adaptor(operands, attributes, properties, regions);
  return hlo::inferReverseOp(location, adaptor.getOperand().getType(),
                             inferredReturnTypes);
}

LogicalResult ReverseOp::reifyReturnTypeShapes(
    OpBuilder& builder, ValueRange operands,
    SmallVectorImpl<Value>& reifiedReturnShapes) {
  return ::mlir::hlo::deriveShapeFromOperand(
      &builder, getOperation(), operands.front(), &reifiedReturnShapes);
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
    ValueShapeRange operands, DictionaryAttr attributes,
    OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  RngOp::Adaptor adaptor(operands, attributes, properties, regions);
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
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange,
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
    ValueShapeRange operands, DictionaryAttr attributes,
    OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  SetDimensionSizeOp::Adaptor adaptor(operands, attributes, properties,
                                      regions);
  return hlo::inferSetDimensionSizeOp(
      getStablehloDialect(context), location, adaptor.getOperand().getType(),
      adaptor.getSize(), adaptor.getDimension(), inferredReturnShapes);
}

mlir::Speculation::Speculatability SetDimensionSizeOp::getSpeculatability() {
  // If the dimension being set is not constant, it is only speculatable if it
  // is dynamic in the output.
  auto resultType = getType();
  if (!matchPattern(getSize(), m_Constant()) &&
      !resultType.isDynamicDim(getDimension()))
    return mlir::Speculation::NotSpeculatable;

  // For all other dimensions, if the dimension is static in the output, it must
  // be static in the input.
  auto inputType = getOperand().getType();
  for (size_t i : llvm::seq(resultType.getRank())) {
    if (i == getDimension()) continue;
    if (!resultType.isDynamicDim(i) && inputType.isDynamicDim(i))
      return mlir::Speculation::NotSpeculatable;
  }
  return mlir::Speculation::Speculatable;
}

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//

LogicalResult TransposeOp::verify() {
  return hlo::verifyTransposeOp(getLoc(), getOperand().getType(),
                                getPermutation(), getResult().getType());
}

//===----------------------------------------------------------------------===//
// PadOp
//===----------------------------------------------------------------------===//

LogicalResult PadOp::inferReturnTypes(
    MLIRContext*, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  PadOp::Adaptor adaptor(operands, attributes, properties, regions);
  return hlo::inferPadOp(location, adaptor.getOperand().getType(),
                         adaptor.getPaddingValue().getType(),
                         adaptor.getEdgePaddingLow(),
                         adaptor.getEdgePaddingHigh(),
                         adaptor.getInteriorPadding(), inferredReturnTypes);
}

LogicalResult PadOp::reifyReturnTypeShapes(
    OpBuilder& builder, ValueRange operands,
    SmallVectorImpl<Value>& reifiedReturnShapes) {
  PadOp::Adaptor adaptor(operands, getOperation()->getAttrDictionary(),
                         getProperties());
  auto loc = this->getLoc();
  Value operand = adaptor.getOperand();
  auto operandTy = cast<RankedTensorType>(operand.getType());

  auto padHigh = adaptor.getEdgePaddingHigh();
  auto padLow = adaptor.getEdgePaddingLow();
  auto padInterior = adaptor.getInteriorPadding();

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

  auto operandType = cast<RankedTensorType>(operand.getType());

  auto loc = this->getLoc();
  SmallVector<Value, 4> shapeValues;
  shapeValues.reserve(operandType.getRank());
  Type shapeScalarType =
      cast<ShapedType>(edgePaddingLow.getType()).getElementType();

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

mlir::Speculation::Speculatability DynamicPadOp::getSpeculatability() {
  return hlo::getShapedSpeculatability(getOperation(), /*shapeCount=*/3);
}

//===----------------------------------------------------------------------===//
// ReshapeOp
//===----------------------------------------------------------------------===//

LogicalResult ReshapeOp::verify() {
  return hlo::verifyReshapeOp(getLoc(), getOperand(), getResult());
}

mlir::Speculation::Speculatability ReshapeOp::getSpeculatability() {
  if (getOperand().getType().hasStaticShape())
    return mlir::Speculation::Speculatable;
  return mlir::Speculation::NotSpeculatable;
}

//===----------------------------------------------------------------------===//
// ReplicaId Op
//===----------------------------------------------------------------------===//

LogicalResult ReplicaIdOp::inferReturnTypes(
    MLIRContext* context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr, OpaqueProperties, RegionRange,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  return hlo::inferReplicaIdOp(context, location, inferredReturnTypes);
}

//===----------------------------------------------------------------------===//
// PartitionId Op
//===----------------------------------------------------------------------===//

LogicalResult PartitionIdOp::inferReturnTypes(
    MLIRContext* context, std::optional<Location> location,
    ValueRange /*operands*/, DictionaryAttr, OpaqueProperties, RegionRange,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  return hlo::inferPartitionIdOp(context, location, inferredReturnTypes);
}

//===----------------------------------------------------------------------===//
// If Op
//===----------------------------------------------------------------------===//

LogicalResult IfOp::inferReturnTypes(
    MLIRContext*, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  IfOp::Adaptor adaptor(operands, attributes, properties, regions);
  return hlo::inferIfOp(location, adaptor.getPred(), adaptor.getRegions(),
                        inferredReturnTypes);
}

//===----------------------------------------------------------------------===//
// Case Op
//===----------------------------------------------------------------------===//

LogicalResult CaseOp::inferReturnTypes(
    MLIRContext*, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  CaseOp::Adaptor adaptor(operands, attributes, properties, regions);
  return hlo::inferCaseOp(location, adaptor.getIndex(), adaptor.getRegions(),
                          inferredReturnTypes);
}

//===----------------------------------------------------------------------===//
// SliceOp
//===----------------------------------------------------------------------===//

LogicalResult SliceOp::inferReturnTypes(
    MLIRContext* context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  SliceOpAdaptor adaptor(operands, attributes, properties);
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
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  SortOp::Adaptor adaptor(operands, attributes, properties, regions);
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

  auto operandType = cast<RankedTensorType>(operand.getType());

  Location loc = this->getLoc();
  SmallVector<int64_t, 4> permutation(this->getPermutation());
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
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  TransposeOp::Adaptor adaptor(operands, attributes, properties, regions);
  return hlo::inferTransposeOp(loc, adaptor.getOperand(),
                               adaptor.getPermutation(), inferredReturnTypes);
}

mlir::Speculation::Speculatability TransposeOp::getSpeculatability() {
  // This is the same logic as SpeculatableIfStaticDimInOutputIsStaticInInput,
  // except it accounts for the permutation.
  auto inputType = getOperand().getType();
  auto resultType = getType();
  auto perm = getPermutation();
  for (size_t i : llvm::seq(resultType.getRank())) {
    if (!resultType.isDynamicDim(i) && inputType.isDynamicDim(perm[i]))
      return mlir::Speculation::NotSpeculatable;
  }
  return mlir::Speculation::Speculatable;
}

//===----------------------------------------------------------------------===//
// TriangularSolveOp
//===----------------------------------------------------------------------===//

LogicalResult TriangularSolveOp::inferReturnTypeComponents(
    MLIRContext*, std::optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  TriangularSolveOp::Adaptor adaptor(operands, attributes, properties, regions);
  bool isTransposeAInvalid =
      (adaptor.getTransposeA() == Transpose::TRANSPOSE_INVALID);
  return hlo::inferTriangularSolveOp(location, adaptor.getA(), adaptor.getB(),
                                     adaptor.getLeftSide(), isTransposeAInvalid,
                                     inferredReturnShapes);
}

mlir::Speculation::Speculatability TriangularSolveOp::getSpeculatability() {
  // If `unit_diagonal` is true, the implementation can assume that the diagonal
  // elements of `a` are equal to 1, which may not be the case at runtime, which
  // may lead to undefined behavior.
  if (getUnitDiagonal()) return mlir::Speculation::NotSpeculatable;

  // If the inputs are statically shaped, they will be fully verified
  // statically. If the inputs are dynamic, then mismatches could occur at
  // runtime.
  auto lhsType = cast<RankedTensorType>(getOperand(0).getType());
  auto rhsType = cast<RankedTensorType>(getOperand(1).getType());
  if (lhsType.hasStaticShape() && rhsType.hasStaticShape())
    return mlir::Speculation::Speculatable;

  return mlir::Speculation::NotSpeculatable;
}

//===----------------------------------------------------------------------===//
// GetTupleElementOp
//===----------------------------------------------------------------------===//

LogicalResult GetTupleElementOp::inferReturnTypes(
    MLIRContext*, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  GetTupleElementOp::Adaptor adaptor(operands, attributes, properties, regions);
  return hlo::inferGetTupleElementOp(location, adaptor.getOperand(),
                                     adaptor.getIndex(), inferredReturnTypes);
}

//===----------------------------------------------------------------------===//
// TupleOp
//===----------------------------------------------------------------------===//

LogicalResult TupleOp::inferReturnTypes(
    MLIRContext* context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  TupleOp::Adaptor adaptor(operands, attributes, properties, regions);
  return hlo::inferTupleOp(context, location, adaptor.getVal(),
                           inferredReturnTypes);
}

//===----------------------------------------------------------------------===//
// CompareOp
//===----------------------------------------------------------------------===//

void CompareOp::build(OpBuilder& builder, OperationState& result, Value lhs,
                      Value rhs, ComparisonDirection comparisonDirection,
                      ComparisonType compareType) {
  ComparisonTypeAttr comparisonTypeAttr;
  if (compareType != ComparisonType::NOTYPE)
    comparisonTypeAttr =
        ComparisonTypeAttr::get(builder.getContext(), compareType);
  build(builder, result, lhs, rhs,
        ComparisonDirectionAttr::get(builder.getContext(), comparisonDirection),
        comparisonTypeAttr);
}

LogicalResult CompareOp::inferReturnTypeComponents(
    MLIRContext* context, std::optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes,
    OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  CompareOp::Adaptor adaptor(operands, attributes, properties, regions);
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
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  SelectAndScatterOp::Adaptor adaptor(operands, attributes, properties,
                                      regions);
  return hlo::inferSelectAndScatterOp(location, adaptor.getOperand(),
                                      adaptor.getScatter(),
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
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  ScatterOp::Adaptor adaptor(operands, attributes, properties, regions);
  return hlo::inferScatterOp(location, adaptor.getInputs(),
                             adaptor.getUpdateComputation(),
                             inferredReturnTypes);
}

LogicalResult ScatterOp::verify() {
  return hlo::verifyScatterOp(
      getLoc(), getInputs(), getScatterIndices(), getUpdates(),
      getScatterDimensionNumbers().getUpdateWindowDims(),
      getScatterDimensionNumbers().getInsertedWindowDims(),
      getScatterDimensionNumbers().getInputBatchingDims(),
      getScatterDimensionNumbers().getScatterIndicesBatchingDims(),
      getScatterDimensionNumbers().getScatterDimsToOperandDims(),
      getScatterDimensionNumbers().getIndexVectorDim(), getUpdateComputation());
}

mlir::Speculation::Speculatability ScatterOp::getSpeculatability() {
  // When unique_indices is true, if the scatter_indices are not unique, the
  // behavior is undefined.
  // A possible improvement would be to check if the scatter_indices are
  // constant and if so, check if they are unique/sorted, and if so do not
  // return NotSpeculatable. However, such a check could be somewhat costly and
  // has unclear ROI.
  if (getUniqueIndices() || getIndicesAreSorted())
    return mlir::Speculation::NotSpeculatable;
  return llvm::all_of(
             this->getOperation()->getOperandTypes(),
             [](Type t) { return cast<RankedTensorType>(t).hasStaticShape(); })
             ? mlir::Speculation::RecursivelySpeculatable
             : mlir::Speculation::NotSpeculatable;
}

//===----------------------------------------------------------------------===//
// WhileOp
//===----------------------------------------------------------------------===//

LogicalResult WhileOp::inferReturnTypes(
    MLIRContext*, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  WhileOp::Adaptor adaptor(operands, attributes, properties, regions);
  return hlo::inferWhileOp(location, adaptor.getOperand(), inferredReturnTypes);
}

LogicalResult WhileOp::verify() {
  return hlo::verifyWhileOp(getLoc(), getOperand(), getCond(), getBody());
}

void WhileOp::print(OpAsmPrinter& p) {
  hlo::printWhileOp(p, getOperation(), getCond(), getBody());
}

ParseResult WhileOp::parse(OpAsmParser& parser, OperationState& result) {
  return hlo::parseWhileOp(parser, result);
}

//===----------------------------------------------------------------------===//
// UniformDequantizeOp
//===----------------------------------------------------------------------===//

LogicalResult UniformDequantizeOp::inferReturnTypeComponents(
    MLIRContext*, std::optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  UniformDequantizeOp::Adaptor adaptor(operands, attributes, properties,
                                       regions);
  return hlo::inferUniformDequantizeOp(location, adaptor.getOperand(),
                                       inferredReturnShapes);
}

//===----------------------------------------------------------------------===//
// UniformQuantizeOp
//===----------------------------------------------------------------------===//

LogicalResult UniformQuantizeOp::verify() {
  return hlo::verifyUniformQuantizeOp(getLoc(), getOperand(), getResult());
}

}  // namespace stablehlo
}  // namespace mlir

using mlir::hlo::parseComplexOpType;
using mlir::hlo::parseCustomCallTarget;
using mlir::hlo::parseDotDimensionNumbers;
using mlir::hlo::parseExponentMantissa;
using mlir::hlo::parsePairwiseOpType;
using mlir::hlo::parseSameOperandsAndResultType;
using mlir::hlo::parseSelectOpType;
using mlir::hlo::parseSliceRanges;
using mlir::hlo::parseTupleOpType;
using mlir::hlo::parseVariadicOperandWithAttribute;
using mlir::hlo::parseVariadicSameOperandsAndResultType;
using mlir::hlo::printComplexOpType;
using mlir::hlo::printCustomCallTarget;
using mlir::hlo::printDotDimensionNumbers;
using mlir::hlo::printExponentMantissa;
using mlir::hlo::printPairwiseOpType;
using mlir::hlo::printSameOperandsAndResultType;
using mlir::hlo::printSelectOpType;
using mlir::hlo::printSliceRanges;
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

  bool isTokenType(Type type) const override { return isa<TokenType>(type); }

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
  StringRef mnemonic;
  Type type;
  auto parseResultOpt = generatedTypeParser(parser, &mnemonic, type);
  if (parseResultOpt.has_value() && succeeded(*parseResultOpt)) return type;
  parser.emitError(parser.getNameLoc())
      << "unknown stablehlo type: " << mnemonic;
  return nullptr;
}

void StablehloDialect::printType(Type type, DialectAsmPrinter& printer) const {
  if (succeeded(generatedTypePrinter(type, printer))) return;
  printer << "<unknown stablehlo type>";
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
  if (auto type_extensions = dyn_cast<TypeExtensionsAttr>(attr)) {
    hlo::printTypeExtensions(cast<hlo::BoundedAttrInterface>(attr), os);
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
              std::make_pair("input_batching_dims", getInputBatchingDims()),
              std::make_pair("scatter_indices_batching_dims",
                             getScatterIndicesBatchingDims()),
              std::make_pair("scatter_dims_to_operand_dims",
                             getScatterDimsToOperandDims()),
              std::make_pair("index_vector_dim", getIndexVectorDim()));
}
Attribute ScatterDimensionNumbersAttr::parse(AsmParser& parser, Type type) {
  if (failed(parser.parseLess())) return {};
  SmallVector<int64_t> updateWindowDims;
  SmallVector<int64_t> insertedWindowDims;
  SmallVector<int64_t> inputBatchingDims;
  SmallVector<int64_t> scatterIndicesBatchingDims;
  SmallVector<int64_t> scatterDimsToOperandDims;
  int64_t indexVectorDim = 0;

  if (failed(parseStruct(
          parser,
          {"update_window_dims", "inserted_window_dims", "input_batching_dims",
           "scatter_indices_batching_dims", "scatter_dims_to_operand_dims",
           "index_vector_dim"},
          {[&]() { return parseDims(parser, updateWindowDims); },
           [&]() { return parseDims(parser, insertedWindowDims); },
           [&]() { return parseDims(parser, inputBatchingDims); },
           [&]() { return parseDims(parser, scatterIndicesBatchingDims); },
           [&]() { return parseDims(parser, scatterDimsToOperandDims); },
           [&]() { return parser.parseInteger(indexVectorDim); }}))) {
    parser.emitError(parser.getCurrentLocation())
        << "failed parsing scatter dimension numbers attribute";
    return {};
  }

  return ScatterDimensionNumbersAttr::get(
      parser.getContext(), updateWindowDims, insertedWindowDims,
      inputBatchingDims, scatterIndicesBatchingDims, scatterDimsToOperandDims,
      indexVectorDim);
}

// Custom printer and parser for GatherDimensionNumbersAttr.
void GatherDimensionNumbersAttr::print(AsmPrinter& printer) const {
  printStruct(printer, "gather", std::make_pair("offset_dims", getOffsetDims()),
              std::make_pair("collapsed_slice_dims", getCollapsedSliceDims()),
              std::make_pair("operand_batching_dims", getOperandBatchingDims()),
              std::make_pair("start_indices_batching_dims",
                             getStartIndicesBatchingDims()),
              std::make_pair("start_index_map", getStartIndexMap()),
              std::make_pair("index_vector_dim", getIndexVectorDim()));
}

Attribute GatherDimensionNumbersAttr::parse(AsmParser& parser, Type type) {
  if (failed(parser.parseLess())) return {};

  SmallVector<int64_t> offsetDims;
  SmallVector<int64_t> collapsedSliceDims;
  SmallVector<int64_t> operandBatchingDims;
  SmallVector<int64_t> startIndicesBatchingDims;
  SmallVector<int64_t> startIndexMap;
  int64_t indexVectorDim = 0;

  if (failed(parseStruct(
          parser,
          {"offset_dims", "collapsed_slice_dims", "operand_batching_dims",
           "start_indices_batching_dims", "start_index_map",
           "index_vector_dim"},
          {[&]() { return parseDims(parser, offsetDims); },
           [&]() { return parseDims(parser, collapsedSliceDims); },
           [&]() { return parseDims(parser, operandBatchingDims); },
           [&]() { return parseDims(parser, startIndicesBatchingDims); },
           [&]() { return parseDims(parser, startIndexMap); },
           [&]() { return parser.parseInteger(indexVectorDim); }}))) {
    parser.emitError(parser.getCurrentLocation())
        << "failed parsing gather dimension numbers attribute";
    return {};
  }

  return GatherDimensionNumbersAttr::get(
      parser.getContext(), offsetDims, collapsedSliceDims, operandBatchingDims,
      startIndicesBatchingDims, startIndexMap, indexVectorDim);
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
void printConvolutionDimensions(AsmPrinter& p,
                                ConvDimensionNumbersAttr dimNums) {
  // TODO(b/202040055): we should check the attribute invariant and print the
  // "raw" form if they are violated, for now report_fatal_error is used to
  // prevent invalid access.
  auto printDim =
      [&p](ArrayRef<int64_t> spatialDims,
           ArrayRef<std::pair<int64_t, NonSpatialDim>> nonSpatialDims) {
        llvm::SmallVector<int64_t> dims(nonSpatialDims.size() +
                                        spatialDims.size());
        // Fill each element of dims with a (< 0) NonSpatialDim enum or a (>=0)
        // spatial dimension index.
        for (const std::pair<int64_t, NonSpatialDim>& nonSpatialDim :
             nonSpatialDims) {
          if (nonSpatialDim.first < 0 ||
              static_cast<size_t>(nonSpatialDim.first) >= dims.size())
            llvm::report_fatal_error("Invalid non-spatial dimension.");
          dims[nonSpatialDim.first] = nonSpatialDim.second;
        }
        for (const auto& spatialDim : llvm::enumerate(spatialDims)) {
          if (spatialDim.value() < 0 ||
              static_cast<size_t>(spatialDim.value()) >= dims.size())
            llvm::report_fatal_error("Invalid spatial dimension.");
          dims[spatialDim.value()] = static_cast<int64_t>(spatialDim.index());
        }

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

  printDim(dimNums.getInputSpatialDimensions(),
           {{dimNums.getInputBatchDimension(), IOBatch},
            {dimNums.getInputFeatureDimension(), IOFeature}});
  p << "x";
  printDim(dimNums.getKernelSpatialDimensions(),
           {{dimNums.getKernelInputFeatureDimension(), KIFeature},
            {dimNums.getKernelOutputFeatureDimension(), KOFeature}});
  p << "->";
  printDim(dimNums.getOutputSpatialDimensions(),
           {{dimNums.getOutputBatchDimension(), IOBatch},
            {dimNums.getOutputFeatureDimension(), IOFeature}});
}

void printConvolutionDimensions(AsmPrinter& p, Operation*,
                                ConvDimensionNumbersAttr dimNums) {
  printConvolutionDimensions(p, dimNums);
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
    AsmParser& parser, ConvDimensionNumbersAttr& dimNums) {
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
  dimNums = ConvDimensionNumbersAttr::get(
      parser.getBuilder().getContext(), inputBatchDimension,
      inputFeatureDimension, inputSpatialDimensions,
      kernelInputFeatureDimension, kernelOutputFeatureDimension,
      kernelSpatialDimensions, outBatchDimension, outputFeatureDimension,
      outputSpatialDimensions);
  return success();
}

ParseResult parseConvolutionDimensions(AsmParser& parser,
                                       ConvDimensionNumbersAttr& dimNums) {
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
  dimNums = ConvDimensionNumbersAttr::get(
      parser.getBuilder().getContext(), inputBatchDimension,
      inputFeatureDimension, inputSpatialDimensions,
      kernelInputFeatureDimension, kernelOutputFeatureDimension,
      kernelSpatialDimensions, outBatchDimension, outputFeatureDimension,
      outputSpatialDimensions);

  return success();
}

Attribute ConvDimensionNumbersAttr::parse(AsmParser& parser, Type type) {
  if (failed(parser.parseLess())) return {};
  ConvDimensionNumbersAttr dimNums;
  if (succeeded(parser.parseOptionalKeyword("raw"))) {
    if (failed(parseConvolutionDimensionsRaw(parser, dimNums))) return {};
    return dimNums;
  }
  if (failed(parseConvolutionDimensions(parser, dimNums))) return {};
  if (failed(parser.parseGreater())) return {};
  return dimNums;
}

namespace {
// Custom formatting for convolution window attributes.
void printWindowPadding(OpAsmPrinter& p, DenseElementsAttr padding) {
  // Padding is Nx2 attribute.
  auto it = padding.value_begin<int64_t>();
  std::vector<std::pair<int64_t, int64_t>> values(padding.getNumElements() / 2);
  for (auto& item : values) {
    int64_t first = *it;
    ++it;
    int64_t second = *it;
    ++it;
    item = {first, second};
  }
  llvm::interleaveComma(values, p, [&](const std::pair<int64_t, int64_t> pair) {
    p << '[' << pair.first << ", " << pair.second << ']';
  });
}
}  // namespace

void printWindowAttributes(OpAsmPrinter& p, Operation* /*op*/,
                           std::optional<DenseI64ArrayAttr> windowStrides,
                           std::optional<DenseIntElementsAttr> padding,
                           std::optional<DenseI64ArrayAttr> lhsDilation,
                           std::optional<DenseI64ArrayAttr> rhsDilation,
                           std::optional<DenseBoolArrayAttr> windowReversal) {
  using pair_t = std::pair<Attribute, StringRef>;
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

  llvm::interleaveComma(nonNullAttributes, p, [&](const pair_t& attr) {
    p << attr.second << " = [";

    if (attr.second == "pad") {
      printWindowPadding(p, dyn_cast<DenseIntElementsAttr>(attr.first));
    } else if (attr.second == "reverse") {
      llvm::interleaveComma(cast<DenseBoolArrayAttr>(attr.first).asArrayRef(),
                            p);
    } else {
      llvm::interleaveComma(cast<DenseI64ArrayAttr>(attr.first).asArrayRef(),
                            p);
    }

    p << ']';
  });
}

ParseResult parseWindowAttributes(OpAsmParser& parser,
                                  DenseI64ArrayAttr& windowStrides,
                                  DenseIntElementsAttr& padding,
                                  DenseI64ArrayAttr& lhsDilation,
                                  DenseI64ArrayAttr& rhsDilation,
                                  DenseBoolArrayAttr& windowReversal) {
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
      if (attributeName == "reverse") {
        auto boolVector =
            llvm::map_to_vector<4>(values, [](int64_t v) { return v != 0; });
        windowReversal =
            DenseBoolArrayAttr::get(parser.getContext(), boolVector);
      } else {
        auto attr = parser.getBuilder().getDenseI64ArrayAttr(values);

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
  OpBuilder::InsertionGuard insertionPointGuard(*builder);

  Location loc = body->getLoc();
  Block* block = builder->createBlock(body);
  // Add two arguments for each element type.
  for (Type elementType : elementTypes) {
    ShapedType shapedType = RankedTensorType::get({}, elementType);
    block->addArguments({shapedType, shapedType},
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
    if (isa<FloatType>(elementType)) {
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
  auto elementsAttr = dyn_cast<ElementsAttr>(value);
  // HLO dialect constants only support ElementsAttr unlike standard dialect
  // constant which supports all attributes.
  if (!elementsAttr) return nullptr;
  // HLO dialect constants require the type of value and result to match.
  if (type != elementsAttr.getType()) return nullptr;

  return builder.create<ConstantOp>(loc, type, elementsAttr);
}

std::optional<StablehloDialectVersion> StablehloDialect::getVersion() const {
  return version;
}

void StablehloDialect::setVersion(
    std::optional<StablehloDialectVersion> version) {
  this->version = version;
}

}  // namespace stablehlo
}  // namespace mlir
