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
#include "stablehlo/dialect/StablehloEnums.cpp.inc"
#define GET_ATTRDEF_CLASSES
#include "stablehlo/dialect/StablehloAttrs.cpp.inc"

namespace mlir {
namespace stablehlo {
namespace {
void createArgs(ArrayRef<OpAsmParser::UnresolvedOperand> operands,
                ArrayRef<Type> types,
                SmallVector<OpAsmParser::Argument>& args) {
  for (auto argAndType : llvm::zip(operands, types)) {
    auto& arg = args.emplace_back();
    arg.ssaName = std::get<0>(argAndType);
    arg.type = std::get<1>(argAndType);
  }
}

// Checks if the vector `nums` has duplicates.
const auto hasDuplicates = [](const ArrayRef<int64_t> nums) {
  llvm::SmallDenseSet<int64_t> set(nums.begin(), nums.end());
  return set.size() != nums.size();
};

//===----------------------------------------------------------------------===//
// Utilities for the canonicalize patterns
//===----------------------------------------------------------------------===//

// Verifies that dimension attribute for the op correctly indexes in operand or
// result shape.
template <typename OpT>
static LogicalResult verifyDimAttr(OpT op) {
  int64_t rank = -1;
  if (auto ty =
          op.getOperand().getType().template dyn_cast<RankedTensorType>()) {
    rank = ty.getRank();
  } else if (auto ty = op.getType().template dyn_cast<RankedTensorType>()) {
    rank = ty.getRank();
  } else {
    return success();
  }

  int64_t dim = op.getDimension();
  if (dim < 0 || dim >= rank)
    return op.emitOpError() << "requires dimension attribute in range [0, "
                            << rank << "); found (" << dim << ")";
  return success();
}

// Common shape function helper for RngNormal and RngUniform.
static LogicalResult rngInferReturnTypeComponents(
    MLIRContext* context, Optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  if (operands.size() != 3)
    return emitOptionalError(location, "expected 3 operands");

  SmallVector<int64_t> shapeVector;
  Value shapeOperand = operands[2];
  auto shapeOperandType = shapeOperand.getType().cast<ShapedType>();
  Type elementType = getElementTypeOrSelf(operands[1]);

  // Operand `shape` (1D by ODS) may be a constant or not, if `shape` is:
  // 1, not constant and have dynimic dim (tensor<?x>): infer tensor<*x>.
  // 2. not constant nor dynimic (e.g. tensor<3xi64>): infer tensor<?x?x?x>.
  // 3. constant (e.g. dense<[2, 3, 5]>): infer tensor<2x3x5x>.

  // Match to check whether the `shape` operand is a constant.
  DenseIntElementsAttr shape;
  if (!matchPattern(shapeOperand, m_Constant(&shape))) {
    int size = shapeOperandType.getDimSize(0);
    if (hlo::isDynamicDimSize(size)) {
      inferredReturnShapes.emplace_back(elementType);
      return success();
    }
    shapeVector.resize(size, ShapedType::kDynamicSize);
    inferredReturnShapes.emplace_back(shapeVector, elementType);
    return success();
  }

  // `shape` operand is a constant.
  shapeVector.reserve(shape.size());
  for (const APInt& fp : shape.getValues<APInt>())
    shapeVector.push_back(fp.getSExtValue());
  inferredReturnShapes.emplace_back(shapeVector, elementType);
  return success();
}

// Returns a new scalar integer value having type `type`. Here `type` must be
// an integer or index type.
Value maybeCastTo(OpBuilder& b, Location loc, Value value, Type type) {
  if (type == value.getType()) return value;
  assert(type.isIndex() || value.getType().isIndex());
  return b.create<arith::IndexCastOp>(loc, type, value);
}

}  // namespace

//===----------------------------------------------------------------------===//
// Utilities for attributes
//===----------------------------------------------------------------------===//

LogicalResult TypeExtensionsAttr::verifyEncoding(
    llvm::ArrayRef<int64_t> bounds, mlir::Type elementType,
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError) const {
  return hlo::verifyBounds(
      getBounds(), RankedTensorType::get(bounds, elementType), emitError);
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

LogicalResult verifyReduceScatter(Operation* op, TypeRange operandTypes,
                                  TypeRange resultTypes,
                                  uint64_t scatterDimension) {
  // If operand and result are both ranked, then the size of the scatter
  // dimension in the operand should be a multiple of the size of the scatter
  // dimension in the result.

  // TODO(zhouxin) Change the ODS definition to return int64_t.
  if (static_cast<int64_t>(scatterDimension) < 0) {
    return op->emitOpError("expects scatter_dimension >= 0");
  }

  for (auto it : llvm::zip(operandTypes, resultTypes)) {
    auto operandType = std::get<0>(it).cast<ShapedType>();
    auto resultType = std::get<1>(it).cast<ShapedType>();
    if (!operandType.hasRank() || !resultType.hasRank()) continue;
    if (operandType.getRank() != resultType.getRank())
      return op->emitOpError() << "operand and result should have same rank";
    if (static_cast<int64_t>(scatterDimension) >= operandType.getRank())
      return op->emitOpError()
             << "scatter dim should be less than operand/result rank";
    if (operandType.isDynamicDim(scatterDimension) ||
        resultType.isDynamicDim(scatterDimension))
      continue;
    if (operandType.getDimSize(scatterDimension) == 0)
      return op->emitOpError() << "operand scatter dimension cannot be zero";
    if (resultType.getDimSize(scatterDimension) == 0)
      return op->emitOpError() << "result scatter dimension cannot be zero";
    if ((operandType.getDimSize(scatterDimension) %
         resultType.getDimSize(scatterDimension)) != 0)
      return op->emitOpError()
             << "operand scatter dimension has size "
             << operandType.getDimSize(scatterDimension)
             << ", expected to be a multiple of result scatter dimension size "
             << resultType.getDimSize(scatterDimension);

    // Non scatter dimensions should be equal.
    for (uint64_t index : llvm::seq<uint64_t>(0, operandType.getRank())) {
      if (index == scatterDimension || operandType.isDynamicDim(index) ||
          resultType.isDynamicDim(index))
        continue;
      if (operandType.getDimSize(index) != resultType.getDimSize(index))
        return op->emitOpError()
               << "non scatter dimensions should be same for operand ("
               << operandType.getDimSize(index) << ") and result ("
               << resultType.getDimSize(index) << ")";
    }
  }
  return success();
}

LogicalResult ReduceScatterOp::verify() {
  if (failed(verifyReplicaGroups(*this, /*is_uniform_sized=*/true)))
    return failure();
  auto operandType = getOperand().getType().cast<TensorType>();
  bool operandTypeRanked = operandType.isa<RankedTensorType>();
  Block& block = getComputation().front();
  SmallVector<TensorType> accumulatorSubshapes;
  if (failed(hlo::verifyReducerShape(
          this->getLoc(), block, {operandType},
          {RankedTensorType::get({}, operandType.getElementType())},
          /*numInputs=*/1, /*allowedDimensions=*/{},
          /*allInputsUnranked=*/!operandTypeRanked, accumulatorSubshapes)))
    return failure();

  return verifyReduceScatter(*this,
                             /*operandTypes=*/{getOperand().getType()},
                             /*resultTypes=*/{getType()},
                             /*scatterDimension=*/getScatterDimension());
}

//===----------------------------------------------------------------------===//
// CompatibleOperandsAndResultType
//===----------------------------------------------------------------------===//

// TODO(b/231358795): Review the use of InferTypeOpInterface for ops that
// support quantization or sparsity.
#define INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(Op)                        \
  LogicalResult Op::inferReturnTypeComponents(                                \
      MLIRContext* context, Optional<Location> location,                      \
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
INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(ReverseOp)
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
// ConstantOp
//===----------------------------------------------------------------------===//

OpFoldResult ConstantOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.empty() && "constant has no operands");

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
    MLIRContext*, Optional<Location>, ValueRange operands,
    DictionaryAttr attributes, RegionRange,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  ConstantOpAdaptor adaptor(operands, attributes);
  Type type = adaptor.getValue().getType();
  inferredReturnTypes.push_back(type);
  return success();
}

bool ConstantOp::isCompatibleReturnTypes(TypeRange l, TypeRange r) {
  if (l.size() != r.size() || l.size() != 1) return false;
  auto lhsTy = l.front().cast<TensorType>();
  auto rhsTy = r.front().cast<TensorType>();
  // For comparisons of the uniform quantized element based tensor type, use the
  // storage type since the constant value will be stored through the underlying
  // storage type.
  if (auto rhsElemTy =
          rhsTy.getElementType().dyn_cast<quant::QuantizedType>()) {
    rhsTy = hlo::getSameShapeTensorType(rhsTy, rhsElemTy.getStorageType());
  }
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
    if (parser.parseType(resultTy)) {
      return failure();
    }
    result.addTypes(resultTy);
    return success();
  }

  ElementsAttr valueAttr;
  if (parser.parseOptionalAttrDict(result.attributes)) return failure();

  if (parser.parseCustomAttributeWithFallback(valueAttr, Type{}, "value",
                                              result.attributes)) {
    return failure();
  }
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
// CustomCallOp
//===----------------------------------------------------------------------===//

LogicalResult CustomCallOp::verify() {
  // If both operand and result layout attributes are not specified then nothing
  // to verify.
  if (!getOperandLayouts().has_value() && !getResultLayouts().has_value())
    return success();

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
      return emitOpError() << "Number of " << valueName
                           << "s must match the number of " << valueName
                           << " layouts, " << types.size()
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

      // For non-tensor types e.g. !stablehlo.token, the layout should be empty.
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
        return emitOpError() << "incorrect layout " << layout << " for type "
                             << type << ", layout must be a permutation of [0, "
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
  // For the common case of a single tuple result packing non-tuple values, the
  // i-th element of `result_layouts` specifies layout for i-th element of the
  // result tuple.
  TypeRange resultTypes;
  if (getNumResults() == 1 && getResult(0).getType().isa<TupleType>())
    resultTypes = getResult(0).getType().cast<TupleType>().getTypes();
  else
    resultTypes = getResultTypes();

  // Verify that operands and operand layouts match.
  if (failed(
          verifyTypesAndLayouts(getOperandTypes(), operandLayouts, "operand")))
    return failure();

  // Verify that results and result layouts match.
  return verifyTypesAndLayouts(resultTypes, resultLayouts, "result");
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

// The following properties are already enforced by the ODS:
//   P0. a.element_type is floating or complex
// We intend to verify the following properties
//   P1. The 'a' argument to Cholesky must have rank >= 2, got shape %s
//   P2. The two minor dimensions of 'a' must have equal size, got %s.
LogicalResult CholeskyOp::inferReturnTypeComponents(
    MLIRContext*, Optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  CholeskyOp::Adaptor adaptor(operands, attributes, regions);
  Type aType = adaptor.getA().getType();
  RankedTensorType aRankedType = aType.dyn_cast<RankedTensorType>();
  if (!aRankedType) {
    inferredReturnShapes.emplace_back(
        aType.cast<TensorType>().getElementType());
    return success();
  }

  ArrayRef<int64_t> aShape = aRankedType.getShape();
  if (aShape.size() < 2) {
    return emitOptionalError(
        location, "argument 'a' must have rank >= 2, got shape ", aShape, ".");
  }

  int64_t lastDim = aShape[aShape.size() - 1];
  int64_t penultimateDim = aShape[aShape.size() - 2];
  if (!hlo::isDynamicDimSize(lastDim) &&
      !hlo::isDynamicDimSize(penultimateDim) && lastDim != penultimateDim) {
    return emitOptionalError(
        location, "minor dimensions of 'a' must have equal size, got shape ",
        aShape, ".");
  }
  inferredReturnShapes.emplace_back(aRankedType.getShape(),
                                    aRankedType.getElementType());
  return success();
}

//===----------------------------------------------------------------------===//
// DotOp
//===----------------------------------------------------------------------===//
namespace {
bool dimCompatible(int64_t a, int64_t b) {
  return hlo::isDynamicDimSize(a) || hlo::isDynamicDimSize(b) || a == b;
}

ShapedType inferDotReturnType(ShapedType lhs, ShapedType rhs) {
  auto elementType = lhs.getElementType();
  if (!lhs.hasRank() || !rhs.hasRank()) {
    return UnrankedTensorType::get(elementType);
  }

  // vector dot vector
  if (1 == lhs.getRank() && 1 == rhs.getRank() &&
      dimCompatible(lhs.getDimSize(0), rhs.getDimSize(0))) {
    return RankedTensorType::get({}, elementType);
  }
  // matrix dot vector
  if (2 == lhs.getRank() && 1 == rhs.getRank() &&
      dimCompatible(lhs.getDimSize(1), rhs.getDimSize(0))) {
    return RankedTensorType::get({lhs.getDimSize(0)}, elementType);
  }
  // vector dot matrix
  if (1 == lhs.getRank() && 2 == rhs.getRank() &&
      dimCompatible(lhs.getDimSize(0), rhs.getDimSize(0))) {
    return RankedTensorType::get({rhs.getDimSize(1)}, elementType);
  }
  // matrix dot matrix
  if (2 == lhs.getRank() && 2 == rhs.getRank() &&
      dimCompatible(lhs.getDimSize(1), rhs.getDimSize(0))) {
    int64_t shape[2] = {lhs.getDimSize(0), rhs.getDimSize(1)};
    return RankedTensorType::get(shape, elementType);
  }
  return {};
}
}  // namespace

LogicalResult DotOp::inferReturnTypes(
    MLIRContext*, Optional<Location>, ValueRange operands, DictionaryAttr,
    RegionRange, SmallVectorImpl<Type>& inferredReturnTypes) {
  DotOp::Adaptor op(operands);
  auto lhsType = op.getLhs().getType().cast<ShapedType>();
  auto rhsType = op.getRhs().getType().cast<ShapedType>();
  inferredReturnTypes.push_back(inferDotReturnType(lhsType, rhsType));
  return success();
}

LogicalResult DotOp::verify() {
  auto lhsType = getLhs().getType().cast<ShapedType>();
  auto rhsType = getRhs().getType().cast<ShapedType>();
  auto resultType = getType().cast<ShapedType>();
  auto expectReturnType = inferDotReturnType(lhsType, rhsType);
  if (!expectReturnType) {
    return emitError() << "Unexpected operands type: " << lhsType << " and "
                       << rhsType;
  }
  if (resultType.hasRank() && expectReturnType.hasRank()) {
    if (resultType.getShape() != expectReturnType.getShape()) {
      return emitError() << "Unexpected result type: has " << resultType
                         << " but inferred " << expectReturnType
                         << " from operands " << lhsType << " and " << rhsType;
    }
  }
  return success();
}

// PrecisionConfig - Optional attribute, print the array as raw enums
//
// {precision_config = [#stablehlo<precision DEFAULT>,
//                      #stablehlo<precision DEFAULT>]}
// ==> ..., precision = [DEFAULT, DEFAULT]
void printPrecisionConfig(OpAsmPrinter& p, Operation*,
                          ::mlir::ArrayAttr attrArr) {
  // Precision config is an optional attribute, passes null if not specified.
  if (!attrArr) return;

  p << ", precision = [";
  llvm::interleaveComma(attrArr, p, [&](Attribute const& attr) {
    p << stringifyPrecision(attr.cast<PrecisionAttr>().getValue());
  });
  p << ']';
}

ParseResult parsePrecisionConfig(OpAsmParser& parser, mlir::ArrayAttr& attr) {
  if (failed(parser.parseOptionalComma())) {
    return success();  // No precision config specified
  }

  if (failed(parser.parseKeyword("precision")) || failed(parser.parseEqual()))
    return failure();

  SmallVector<Attribute> attrs;
  if (failed(parser.parseCommaSeparatedList(
          AsmParser::Delimiter::Square, [&]() -> ParseResult {
            attrs.push_back(PrecisionAttr::parse(parser, {}));
            return success(/*isSuccess=*/static_cast<bool>(attrs.back()));
          }))) {
    return failure();
  }

  attr = mlir::ArrayAttr::get(parser.getContext(), attrs);
  return success();
}

//===----------------------------------------------------------------------===//
// DotGeneralOp
//===----------------------------------------------------------------------===//

LogicalResult DotGeneralOp::inferReturnTypeComponents(
    MLIRContext*, Optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  DotGeneralOp::Adaptor adaptor(operands, attributes, regions);
  return hlo::inferDotGeneralOp(
      location, adaptor.getLhs(), adaptor.getRhs(),
      adaptor.getDotDimensionNumbersAttr().getLhsBatchingDimensions(),
      adaptor.getDotDimensionNumbersAttr().getRhsBatchingDimensions(),
      adaptor.getDotDimensionNumbersAttr().getLhsContractingDimensions(),
      adaptor.getDotDimensionNumbersAttr().getRhsContractingDimensions(),
      inferredReturnShapes);
}

LogicalResult DotGeneralOp::reifyReturnTypeShapes(
    OpBuilder& builder, ValueRange operands,
    SmallVectorImpl<Value>& reifiedReturnShapes) {
  auto lhsType = getLhs().getType().dyn_cast<ShapedType>();
  auto rhsType = getRhs().getType().dyn_cast<ShapedType>();
  if (!lhsType || !rhsType) {
    return failure();
  }

  Adaptor adaptor(operands);
  auto dimNumbers = getDotDimensionNumbers();
  SmallVector<Value> dimensions;
  for (const int64_t lhsDim : dimNumbers.getLhsBatchingDimensions()) {
    dimensions.push_back(
        builder.create<tensor::DimOp>(getLoc(), adaptor.getLhs(), lhsDim));
  }

  for (int64_t i = 0; i < lhsType.getRank(); i++) {
    if (!llvm::is_contained(dimNumbers.getLhsContractingDimensions(), i) &&
        !llvm::is_contained(dimNumbers.getLhsBatchingDimensions(), i)) {
      dimensions.push_back(
          builder.create<tensor::DimOp>(getLoc(), adaptor.getLhs(), i));
    }
  }
  for (int64_t i = 0; i < rhsType.getRank(); i++) {
    if (!llvm::is_contained(dimNumbers.getRhsContractingDimensions(), i) &&
        !llvm::is_contained(dimNumbers.getRhsBatchingDimensions(), i)) {
      dimensions.push_back(
          builder.create<tensor::DimOp>(getLoc(), adaptor.getRhs(), i));
    }
  }

  reifiedReturnShapes.push_back(
      builder.create<tensor::FromElementsOp>(getLoc(), dimensions));
  return success();
}

//===----------------------------------------------------------------------===//
// FftOp
//===----------------------------------------------------------------------===//

// We intend to verify the following properties
// P1. 1 <= rank <= 3
// P2. Element types agree with fft_type
// P3. Operand shape dimensions agree with fft_length for the given fft_type
LogicalResult FftOp::inferReturnTypeComponents(
    MLIRContext*, Optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  FftOp::Adaptor adaptor(operands, attributes, regions);
  auto fftLength = adaptor.getFftLength().getValues<int64_t>();
  int64_t fftRank = fftLength.size();

  // P1.
  if (fftRank > 3 || fftRank < 1) {
    return emitOptionalError(location, "rank must be between 1 and 3, but got ",
                             fftRank, ".");
  }

  // P2. Element type agreement
  // FFT : C -> C
  // IFFT : C -> C
  // RFFT : R -> C
  // IRFFT : C -> R
  auto fftType = adaptor.getFftType();
  auto operandType = adaptor.getOperand().getType().cast<TensorType>();
  Type operandElementType = operandType.getElementType();
  // Check the input element type and infer return element type
  if (fftType == FftType::RFFT) {
    if (!operandElementType.isF32() && !operandElementType.isF64()) {
      return emitOptionalError(
          location, "RFFT requires f32 or f64 input type, but is given ",
          operandElementType, ".");
    }
  } else {
    if (!operandElementType.isa<ComplexType>()) {
      return emitOptionalError(
          location, stringifyFftType(fftType),
          " takes a complex tensor as input, but is given ", operandType, ".");
    }
  }
  // Generate the output element type
  Type resultElementType = operandElementType;
  if (fftType == FftType::RFFT) {  // RFFT : R -> C
    resultElementType = ComplexType::get(resultElementType);
  } else if (fftType == FftType::IRFFT) {  // IRFFT : C -> R
    resultElementType = operandElementType.cast<ComplexType>().getElementType();
  }

  // P3. Check input shape and infer return shape
  operandType = operandType.dyn_cast<RankedTensorType>();
  if (!operandType) {
    inferredReturnShapes.emplace_back(resultElementType);
    return success();
  }
  auto operandShape = operandType.getShape();
  if (static_cast<int64_t>(operandShape.size()) < fftRank) {
    return emitOptionalError(
        location, "operand rank must not be less than fft rank of ", fftRank,
        " for operand of type ", operandType, ".");
  }

  SmallVector<int64_t> resultShape = to_vector(operandShape);

  if (fftType == FftType::RFFT) {
    auto shapeBack = operandShape.take_back(fftRank);
    for (auto [operandDim, fftDim] : llvm::zip(shapeBack, fftLength)) {
      if (operandDim != fftDim) {
        return emitOptionalError(
            location,
            "RFFT requires innermost dimensions match fft_length. Got: ",
            operandShape, " but wanted ", fftLength, ".");
      }
    }
    if (fftLength[fftRank - 1] != 0) {
      resultShape[resultShape.size() - 1] = fftLength[fftRank - 1] / 2 + 1;
    }
  }
  if (fftType == FftType::IRFFT) {
    auto shapeBack = operandShape.take_back(fftRank).drop_back();
    for (auto [operandDim, fftDim] : llvm::zip(shapeBack, fftLength)) {
      if (operandDim != fftDim) {
        return emitOptionalError(location,
                                 "IRFFT requires non-final dimensions "
                                 "match fft_length. Got: ",
                                 operandShape, " but wanted ", fftLength,
                                 ", and ", operandDim, " != ", fftDim, ".");
      }
    }
    if ((operandShape[operandShape.size() - 1] != 0 ||
         fftLength[fftRank - 1] != 0) &&
        operandShape[operandShape.size() - 1] != fftLength[fftRank - 1] / 2 + 1)
      return emitOptionalError(location,
                               "IRFFT requires innermost dimension match "
                               "fft_length[-1]/2+1. Got: ",
                               operandShape, " but fft_length is ", fftLength,
                               ".");
    resultShape[resultShape.size() - 1] = fftLength[fftRank - 1];
  }

  inferredReturnShapes.emplace_back(resultShape, resultElementType);
  return success();
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
  for (int64_t val : gather->getSliceSizes().getValues<int64_t>()) {
    sliceSizes.push_back(builder.create<arith::ConstantIndexOp>(loc, val));
  }
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

// Verify the following properties:
//  P1. Verify no repeat in start_index_map.
//  P2. Verify 0 <= start_index_map[i] < rank(operand), for every i.
//  P3. Verify 0 <= index_vector_dim <= rank(start_indices).
//  P4. Verify size(start_index_map) == shape(start_indices)[index_vector_dim].
//  P5. Verify offset_dims is_sorted and not repeated.
//  P6. Verify collapsed_slice_dims is_sorted and not repeated.
//  P7. Verify rank(operand) == size(offset_dims) + size(collapsed_slice_dims).
//  P8. Verify slice_sizes has rank of 1.
//  P9. Verify size(slice_sizes) == rank(operand).
//  P10. Verify 0 <= collapsed_slice_dims[i] < size(slice_sizes) for all items.
static LogicalResult verifyGather(
    ShapeAdaptor operandShape, ShapeAdaptor startIndicesShape,
    ShapeAdaptor sliceSizesShape, GatherDimensionNumbersAttr dimensionNumbers,
    llvm::function_ref<InFlightDiagnostic()> errorEmitter) {
  int64_t indexVectorDim = dimensionNumbers.getIndexVectorDim();

  // Check startIndexMap
  auto startIndexMap = to_vector(dimensionNumbers.getStartIndexMap());
  // P1.
  if (hasDuplicates(startIndexMap))
    return errorEmitter() << "expects start_index_map to not repeat, got: ["
                          << startIndexMap << "]";

  // P2.
  for (int64_t i = 0; i < static_cast<int64_t>(startIndexMap.size()); ++i)
    if (startIndexMap[i] < 0 ||
        (operandShape.hasRank() && startIndexMap[i] >= operandShape.getRank()))
      return errorEmitter()
             << "start_index_map[" << i << "]: " << startIndexMap[i]
             << " is out of bounds for "
             << "operand rank " << operandShape.getRank();

  if (startIndicesShape.hasRank()) {
    // P3.
    // index_vector_dim == start_indices.rank implies a trailing 1 on the shape
    // of start_indices.
    if (indexVectorDim > startIndicesShape.getRank() || indexVectorDim < 0)
      return errorEmitter() << "index_vector_dim " << indexVectorDim
                            << " is out of bounds for start indices with rank "
                            << startIndicesShape.getRank();

    bool impliedTrailingDim = indexVectorDim == startIndicesShape.getRank();
    if (impliedTrailingDim || !startIndicesShape.isDynamicDim(indexVectorDim)) {
      int64_t effectiveDimSize;
      if (impliedTrailingDim)
        effectiveDimSize = 1;
      else
        effectiveDimSize = startIndicesShape.getDimSize(indexVectorDim);
      // P4.
      if (effectiveDimSize !=
          static_cast<int64_t>(dimensionNumbers.getStartIndexMap().size()))
        return errorEmitter() << "start_index_map size ("
                              << dimensionNumbers.getStartIndexMap().size()
                              << ") is not equal to size of index dimension ("
                              << indexVectorDim << ") of start_indices ("
                              << effectiveDimSize << ")";
    }
  }

  // P5.
  auto offsetDims = to_vector(dimensionNumbers.getOffsetDims());
  if (!llvm::is_sorted(offsetDims))
    return errorEmitter() << "expects offset_dims to be sorted, got: ["
                          << offsetDims << "]";
  if (hasDuplicates(offsetDims))
    return errorEmitter() << "expects offset_dims to not repeat, got: ["
                          << offsetDims << "]";

  // P6.
  auto collapsedSliceDims = to_vector(dimensionNumbers.getCollapsedSliceDims());
  if (!llvm::is_sorted(collapsedSliceDims))
    return errorEmitter() << "expects collapsed_slice_dims to be sorted, got: ["
                          << collapsedSliceDims << "]";
  if (hasDuplicates(collapsedSliceDims))
    return errorEmitter()
           << "expects collapsed_slice_dims to not repeat, got: ["
           << collapsedSliceDims << "]";

  // P7.
  int64_t impliedOperandRank = dimensionNumbers.getOffsetDims().size() +
                               dimensionNumbers.getCollapsedSliceDims().size();
  if (operandShape.hasRank() && operandShape.getRank() != impliedOperandRank)
    return errorEmitter() << "offset_dims size ("
                          << dimensionNumbers.getOffsetDims().size()
                          << ") plus collapse_slice_dims size ("
                          << dimensionNumbers.getCollapsedSliceDims().size()
                          << ") is not equal to operand rank ("
                          << operandShape.getRank() << ")";

  // P8.
  // This should be fully expressible with type constraints, but it isn't
  // obvious how to do that with the current infrastructure.
  if (sliceSizesShape.hasRank() && sliceSizesShape.getRank() != 1)
    return errorEmitter() << "slice_sizes.rank != 1";
  if (sliceSizesShape.hasStaticShape()) {
    int64_t sliceSize = sliceSizesShape.getNumElements();

    // P9.
    if (sliceSize != impliedOperandRank)
      return errorEmitter() << "slice_sizes size (" << sliceSize
                            << ") not equal to (implied) operand rank ("
                            << impliedOperandRank << ")";

    // P10.
    for (auto dim : dimensionNumbers.getCollapsedSliceDims())
      if (dim < 0 || dim >= sliceSize)
        return errorEmitter() << "collapsed dimension " << dim
                              << " is out of bounds for slice_sizes.size ("
                              << sliceSize << ")";
  }

  return success();
}

// Verify the following properties:
//  P1. Verifications by verifyGather().
//  P2. Verify slice_sizes[i] <= 1 for i in collapsed_slice_dims.
//  P3. Verify 0 <= slice_sizes[i] < shape(operand)[i], for every i.
static LogicalResult verifyStaticGather(
    ShapeAdaptor operandShape, ShapeAdaptor startIndicesShape,
    DenseIntElementsAttr sliceSizes,
    GatherDimensionNumbersAttr dimensionNumbers,
    llvm::function_ref<InFlightDiagnostic()> errorEmitter) {
  // P1.
  // For some reason the getType call is necessary here
  if (failed(verifyGather(
          /*operandShape=*/operandShape,
          /*startIndicesShape=*/startIndicesShape,
          /*sliceSizesShape=*/sliceSizes.getType(), dimensionNumbers,
          errorEmitter)))
    return failure();

  // P2.
  for (auto dim : dimensionNumbers.getCollapsedSliceDims()) {
    int64_t sliceDimSize = sliceSizes.getValues<int64_t>()[dim];
    if (sliceDimSize > 1) {
      return errorEmitter() << "slice_sizes collapsed dimension " << dim
                            << " should <= 1 but got " << sliceDimSize;
    }
  }

  // P3.
  if (operandShape.hasRank()) {
    for (const auto& it : llvm::enumerate(sliceSizes.getValues<int64_t>())) {
      if (operandShape.isDynamicDim(it.index())) continue;
      auto operandDimSize = operandShape.getDimSize(it.index());
      auto sliceDimSize = it.value();
      if (sliceDimSize < 0 || sliceDimSize > operandDimSize)
        return errorEmitter() << "slice size (" << sliceDimSize
                              << ") is out of bounds for operand dimension ("
                              << operandDimSize << ") at index " << it.index();
    }
  }
  return success();
}

template <typename dimTy>
static void inferGatherShape(
    int64_t resultRank, llvm::function_ref<dimTy(int64_t)> getStartIndicesDim,
    llvm::function_ref<dimTy(int64_t)> getSliceDim,
    GatherDimensionNumbersAttr dimensionNumbers,
    SmallVectorImpl<dimTy>& shape) {
  ArrayRef<int64_t> collapsedSliceDims =
      dimensionNumbers.getCollapsedSliceDims();
  int64_t indexVectorDim = dimensionNumbers.getIndexVectorDim();

  // We don't necessarily know the rank of sliceSizes, but we do know that it
  // can't be larger than the highest collapsed dimension. So go through those
  // and populate the leading dimensions of adjustedSliceSizes. The trailing
  // dimensions can just be adjusted by an offset.
  const auto* maxCollapsedDimIt =
      std::max_element(collapsedSliceDims.begin(), collapsedSliceDims.end());
  int64_t maxCollapsedDim = -1;
  if (maxCollapsedDimIt != collapsedSliceDims.end())
    maxCollapsedDim = *maxCollapsedDimIt;

  SmallVector<dimTy> adjustedSliceSizePrefix;
  for (int dimIndex = 0; dimIndex <= maxCollapsedDim; ++dimIndex) {
    if (llvm::is_contained(collapsedSliceDims, dimIndex)) continue;
    adjustedSliceSizePrefix.push_back(getSliceDim(dimIndex));
  }
  auto getAdjustedSliceDim = [&](int64_t index) -> dimTy {
    if (index < static_cast<int64_t>(adjustedSliceSizePrefix.size()))
      return adjustedSliceSizePrefix[index];
    return getSliceDim(index + collapsedSliceDims.size());
  };

  ArrayRef<int64_t> offsetDims = dimensionNumbers.getOffsetDims();

  // Dimensions in the output that aren't offset dimensions are called batch
  // dimensions.
  SmallVector<int64_t> batchDims;
  for (int dim = 0; dim < resultRank; ++dim)
    if (!llvm::is_contained(offsetDims, dim)) batchDims.push_back(dim);

  for (int i = 0; i < resultRank; ++i) {
    const auto* offsetDimsIt =
        std::find(offsetDims.begin(), offsetDims.end(), i);
    if (offsetDimsIt != offsetDims.end()) {
      auto index = std::distance(offsetDims.begin(), offsetDimsIt);
      shape.push_back(getAdjustedSliceDim(index));
      continue;
    }
    auto* batchDimsIt = std::find(batchDims.begin(), batchDims.end(), i);
    assert(batchDimsIt != batchDims.end());
    auto index = std::distance(batchDims.begin(), batchDimsIt);
    // This can never run into the special case where start_indices gets
    // implicitly expanded with a trailing 1 if
    // index_vector_dim = start_indices.rank because then index would equal
    // index_vector_dim, which means we'd be looking at index+1, which would be
    // out of bounds anyway.
    if (index >= indexVectorDim) ++index;
    shape.push_back(getStartIndicesDim(index));
  }
}

// Verify the following properties:
//  P1. Verify 0 <= offset_dims[i] < output_shape_rank, for every i.
//      (output_shape_rank = size(offset_dims) + rank(start_indices) -1)
static LogicalResult inferGatherReturnTypeComponents(
    ShapeAdaptor operandShape, ShapeAdaptor startIndicesShape,
    llvm::function_ref<int64_t(int64_t)> getSliceDim,
    GatherDimensionNumbersAttr dimensionNumbers,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes,
    llvm::function_ref<InFlightDiagnostic()> errorEmitter) {
  Type elementType = operandShape.getElementType();

  // We need this to determine the result rank. We could still place bounds on
  // the result rank if that was something ShapedTypeComponents could express.
  if (!startIndicesShape.hasRank()) {
    inferredReturnShapes.push_back(elementType);
    return success();
  }

  ArrayRef<int64_t> offsetDims = dimensionNumbers.getOffsetDims();
  int64_t startIndicesRank = startIndicesShape.getRank();
  // If index_vector_dim == start_indices.rank, then an implicit trailing 1 is
  // appended to start_indices shape.
  if (dimensionNumbers.getIndexVectorDim() == startIndicesRank)
    ++startIndicesRank;
  int64_t resultRank = offsetDims.size() + startIndicesRank - 1;
  // P1.
  for (int64_t i = 0; i < static_cast<int64_t>(offsetDims.size()); ++i)
    if (offsetDims[i] < 0 || offsetDims[i] >= resultRank)
      return errorEmitter() << "offset_dims[" << i << "]: " << offsetDims[i]
                            << " is out of bounds for "
                            << "implied result rank " << resultRank;

  auto getStartIndicesDim = [&](int64_t index) {
    return startIndicesShape.getDimSize(index);
  };

  SmallVector<int64_t> shape;
  inferGatherShape<int64_t>(resultRank, getStartIndicesDim, getSliceDim,
                            dimensionNumbers, shape);

  inferredReturnShapes.emplace_back(shape, elementType);
  return success();
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
  Type shapeElTy = startIndices.getType().cast<ShapedType>().getElementType();
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
  inferGatherShape<Value>(resultRank, getStartIndicesDim, getSliceDim,
                          op->getDimensionNumbers(), shapeValues);

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

// The following properties are already enforced by the ODS:
//  P0. Verify the start_indices has element type of integer.
// Verify the following properties:
//  Verifications by verifyStaticGather() and verifyGather() inside it.
//  Verifications by inferGatherReturnTypeComponents.
LogicalResult GatherOp::inferReturnTypeComponents(
    MLIRContext* context, Optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  Location loc = location.value_or(UnknownLoc::get(context));
  auto errorEmitter = [&loc]() {
    return mlir::emitError(loc)
           << "'" << GatherOp::getOperationName() << "' op ";
  };
  GatherOp::Adaptor adaptor(operands, attributes, regions);
  if (failed(adaptor.verify(loc))) return failure();

  // We want the ShapeAdaptors, so can't route via the adaptor :-/
  ShapeAdaptor operandShape = operands.getShape(0);
  ShapeAdaptor startIndicesShape = operands.getShape(1);
  GatherDimensionNumbersAttr dimensionNumbers = adaptor.getDimensionNumbers();
  DenseIntElementsAttr sliceSizesAttr = adaptor.getSliceSizes();

  if (failed(verifyStaticGather(/*operandShape=*/operandShape,
                                /*startIndicesShape=*/startIndicesShape,
                                /*sliceSizes=*/sliceSizesAttr, dimensionNumbers,
                                errorEmitter)))
    return failure();

  auto getSliceDim = [&sliceSizesAttr](int64_t index) -> int64_t {
    return sliceSizesAttr.getValues<int64_t>()[index];
  };

  return inferGatherReturnTypeComponents(operandShape, startIndicesShape,
                                         getSliceDim, dimensionNumbers,
                                         inferredReturnShapes, errorEmitter);
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
    MLIRContext* context, Optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  // This can get called before other op verify methods, so we have to do a
  // bunch of verification up front. With a better story for ordering and/or
  // multi-phase op verification, this should hopefully all go away.
  Location loc = location.value_or(UnknownLoc::get(context));
  auto errorEmitter = [&loc]() {
    return mlir::emitError(loc)
           << "'" << DynamicGatherOp::getOperationName() << "' op ";
  };
  DynamicGatherOp::Adaptor adaptor(operands, attributes, regions);
  if (failed(adaptor.verify(loc))) return failure();

  // We want the ShapeAdaptors, so can't route via the adaptor :-/
  ShapeAdaptor operandShape = operands.getShape(0);
  ShapeAdaptor startIndicesShape = operands.getShape(1);
  ShapeAdaptor sliceSizesShape = operands.getShape(2);
  GatherDimensionNumbersAttr dimensionNumbers = adaptor.getDimensionNumbers();

  if (failed(verifyGather(/*operandShape=*/operandShape,
                          /*startIndicesShape=*/startIndicesShape,
                          /*sliceSizesShape=*/sliceSizesShape, dimensionNumbers,
                          errorEmitter)))
    return failure();

  auto getSliceDim = [](int64_t index) { return ShapedType::kDynamicSize; };
  return inferGatherReturnTypeComponents(operandShape, startIndicesShape,
                                         getSliceDim, dimensionNumbers,
                                         inferredReturnShapes, errorEmitter);
}

//===----------------------------------------------------------------------===//
// GetDimensionSizeOp
//===----------------------------------------------------------------------===//
//
LogicalResult GetDimensionSizeOp::verify() { return verifyDimAttr(*this); }

//===----------------------------------------------------------------------===//
// IotaOp
//===----------------------------------------------------------------------===//

LogicalResult IotaOp::verify() {
  auto shape = getType().cast<ShapedType>();
  if (!shape.hasRank()) return success();

  if (shape.getRank() == 0) return emitOpError() << "does not support scalars.";

  auto iotaDimension = static_cast<int64_t>(this->getIotaDimension());
  if (iotaDimension >= shape.getRank() || iotaDimension < 0)
    return emitOpError()
           << "iota dimension cannot go beyond the output rank or be negative.";
  return success();
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

LogicalResult DynamicUpdateSliceOp::verify() {
  OperandRange indices = getStartIndices();
  if (indices.size() <= 1) return success();

  // Note: start_indices is constrained to Variadic<HLO_ScalarIntTensor>, so it
  // is OK to cast indices to ShapedType here.
  auto idxTensor = indices.take_front().front().getType().cast<ShapedType>();
  Type firstElemTy = idxTensor.getElementType();
  Type elemTy;

  for (auto idx : llvm::drop_begin(indices, 1)) {
    idxTensor = idx.getType().cast<ShapedType>();
    elemTy = idxTensor.getElementType();

    if (firstElemTy != elemTy) {
      return emitOpError() << "start indices must have same element type "
                              "(encountered mismatch: "
                           << firstElemTy << " vs " << elemTy << ")";
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// AbsOp
//===----------------------------------------------------------------------===//

LogicalResult AbsOp::inferReturnTypes(
    MLIRContext*, Optional<Location>, ValueRange operands, DictionaryAttr,
    RegionRange, SmallVectorImpl<Type>& inferredReturnTypes) {
  auto operandTy = (*operands.begin()).getType().cast<ShapedType>();
  Type elementTy = operandTy.getElementType();
  if (auto complexTy = elementTy.dyn_cast<ComplexType>()) {
    elementTy = complexTy.getElementType();
  }

  Type resultTy;
  if (auto rankedOperandTy = operandTy.dyn_cast<RankedTensorType>()) {
    resultTy = RankedTensorType::get(operandTy.getShape(), elementTy,
                                     rankedOperandTy.getEncoding());
  } else if (operandTy.hasRank()) {
    resultTy = RankedTensorType::get(operandTy.getShape(), elementTy);
  } else {
    resultTy = UnrankedTensorType::get(elementTy);
  }
  inferredReturnTypes.push_back(resultTy);
  return success();
}

//===----------------------------------------------------------------------===//
// CollectivePermuteOp
//===----------------------------------------------------------------------===//

// Verifies the source target pairs attached to collective permute.
LogicalResult verifyCollectivePermuteSourceTargetPairs(
    Operation* op, DenseIntElementsAttr attr) {
  auto type = attr.getType().dyn_cast<RankedTensorType>();
  if (type.getRank() != 2)
    return op->emitError() << "expect source_target_pairs attribute to be of "
                              "rank 2, but got rank "
                           << type.getRank();
  if (type.getShape()[1] != 2)
    return op->emitError()
           << "expect source_target_pairs attribute of shape (N, 2), but got ("
           << type.getShape() << ")";
  // Check source target pairs for duplicate sources or targets.
  llvm::DenseSet<int64_t> sources;
  llvm::DenseSet<int64_t> targets;
  for (auto i = attr.begin(), e = attr.end(); i != e; ++i) {
    auto val = (*i).getSExtValue();
    if (i.getIndex() % 2 == 0) {
      bool isUnique = sources.insert(val).second;
      if (!isUnique) return op->emitError() << "duplicate sources not allowed.";
    } else {
      bool isUnique = targets.insert(val).second;
      if (!isUnique) return op->emitError() << "duplicate targets not allowed.";
    }
  }
  return success();
}

LogicalResult CollectivePermuteOp::verify() {
  return verifyCollectivePermuteSourceTargetPairs(*this,
                                                  getSourceTargetPairs());
}

//===----------------------------------------------------------------------===//
// ConvolutionOp
//===----------------------------------------------------------------------===//

namespace {
// Checks:
//  P1. Same sizes for input, kernel and output spatialDims.
//  P2. Spatial and non-spatial dimentions (for input,kernel, &output) should
//      be unique and in range [0, num_dims), where num_dims = rank of input
//      (lhs/rhs) tensors.
//
//  Note that the spatial + non-spatial dimensions may not cover all the
//  dimensions in the range [0,num) because of the presence of 'unknown'
//  dimensions (ref. `printConvolutionDimensions()`)
LogicalResult isSpatialDimensionsValid(
    Value lhs,
    // clang-format off
    int64_t inputBatchDimension,
    int64_t inputFeatureDimension,
    ArrayRef<int64_t> inputSpatialDimensions,
    int64_t kernelInputFeatureDimension,
    int64_t kernelOutputFeatureDimension,
    ArrayRef<int64_t> kernelSpatialDimensions,
    int64_t outputBatchDimension,
    int64_t outputFeatureDimension,
    ArrayRef<int64_t> outputSpatialDimensions,
    // clang-format on
    Optional<Location> location) {
  uint64_t spatialDimNum = inputSpatialDimensions.size();
  // P1.
  if ((spatialDimNum != kernelSpatialDimensions.size()) ||
      (spatialDimNum != outputSpatialDimensions.size()))
    return emitOptionalError(location,
                             "expects the same size for input, kernel "
                             "and output spatial-dimensions, but got ",
                             spatialDimNum, ", ",
                             kernelSpatialDimensions.size(), ", and ",
                             outputSpatialDimensions.size(), " resp.");

  // P2.
  SmallVector<int64_t> inputDimNums(spatialDimNum + 2);
  inputDimNums[0] = inputBatchDimension;
  inputDimNums[1] = inputFeatureDimension;
  std::copy(inputSpatialDimensions.begin(), inputSpatialDimensions.end(),
            inputDimNums.begin() + 2);

  SmallVector<int64_t> windowDimNums(spatialDimNum + 2);
  windowDimNums[0] = kernelInputFeatureDimension;
  windowDimNums[1] = kernelOutputFeatureDimension;
  std::copy(kernelSpatialDimensions.begin(), kernelSpatialDimensions.end(),
            windowDimNums.begin() + 2);

  SmallVector<int64_t> OutputDimNums(spatialDimNum + 2);
  OutputDimNums[0] = outputBatchDimension;
  OutputDimNums[1] = outputFeatureDimension;
  std::copy(outputSpatialDimensions.begin(), outputSpatialDimensions.end(),
            OutputDimNums.begin() + 2);

  auto numDims = lhs.getType().cast<RankedTensorType>().getRank();
  const auto inRange = [numDims](int64_t i) { return 0 <= i && i < numDims; };

  if (!llvm::all_of(inputDimNums, inRange) ||
      !llvm::all_of(windowDimNums, inRange) ||
      !llvm::all_of(OutputDimNums, inRange))
    return emitOptionalError(location,
                             "expects input, kernel, and output "
                             "dimension-numbers to be in-range [0, ",
                             numDims, ").");

  if (hasDuplicates(inputDimNums))
    return emitOptionalError(
        location, "expects input dimension-numbers to be unique, got {",
        inputDimNums, "}.");

  if (hasDuplicates(windowDimNums))
    return emitOptionalError(
        location, "expects kernel dimension-numbers to be unique, got {",
        windowDimNums, "}.");

  if (hasDuplicates(OutputDimNums))
    return emitOptionalError(
        location, "expects output dimension-numbers to be unique, got {",
        OutputDimNums, "}.");

  return success();
}

// Verifies the following properties:
//  P1. The input, kernel, and output spatial-dimentions are valid.
//  P2. Given,
//          input-dimensions: b * input-spatial-dims * f
//          kernel-dimensions: kernel-spatial-dims * i * o
//          output-dimensions: b' * out-spatial-dims * f'
//            where b = input-batch-dims
//            where f = input-feature-dims
//            where i = kernel-input-feature-dims
//            where o = kernel-output-feature-dims
//            where b' = output-batch-dims
//            where f' = output-feature-dims
//      Check the following properties w.r.t feature_group_count (fgc) and
//      batch_group_count (bgc).
//        fgc > 0, bgc > 1 and !(fgc > 1 && bgc > 1)
//        b % bgc == 0
//        f % fgc == 0 and i = f / fgc
//        o (or f') % bgc == 0 and o (or f') % fgc == 0
LogicalResult verifyConvolutionAttributes(
    // clang-format off
    Value lhs, Value rhs,
    int64_t inputBatchDimension,
    int64_t inputFeatureDimension,
    ArrayRef<int64_t> inputSpatialDimensions,
    int64_t kernelInputFeatureDimension,
    int64_t kernelOutputFeatureDimension,
    ArrayRef<int64_t> kernelSpatialDimensions,
    int64_t outputBatchDimension,
    int64_t outputFeatureDimension,
    ArrayRef<int64_t> outputSpatialDimensions,
    int64_t featureGroupCount,
    int64_t batchGroupCount,
    // clang-format on
    Optional<Location> location) {
  // P1.
  if (failed(isSpatialDimensionsValid(
          lhs,
          // clang-format off
          inputBatchDimension,
          inputFeatureDimension,
          inputSpatialDimensions,
          kernelInputFeatureDimension,
          kernelOutputFeatureDimension,
          kernelSpatialDimensions,
          outputBatchDimension,
          outputFeatureDimension,
          outputSpatialDimensions,
          location
          // clang-format on
          )))
    return failure();

  // P2.
  if (featureGroupCount <= 0)
    return emitOptionalError(
        location, "expects feature_group_count to be a positive number, got ",
        featureGroupCount, ".");

  if (batchGroupCount <= 0)
    return emitOptionalError(
        location, "expects batch_group_count to be a positive number, got ",
        batchGroupCount, ".");

  if (batchGroupCount > 1 && featureGroupCount > 1)
    return emitOptionalError(
        location,
        "expects batch_group_count and feature_group_count not to be both "
        "greater than 1. Got ",
        batchGroupCount, " and ", featureGroupCount, " resp.");

  auto lhsType = lhs.getType().cast<RankedTensorType>();
  const int64_t inputFeatures = lhsType.getShape()[inputFeatureDimension];
  const int64_t inputBatch = lhsType.getShape()[inputBatchDimension];

  auto rhsType = rhs.getType().cast<RankedTensorType>();
  const int64_t kernelInputFeatures =
      rhsType.getShape()[kernelInputFeatureDimension];
  const int64_t kernelOutputFeatures =
      rhsType.getShape()[kernelOutputFeatureDimension];

  if (!hlo::isDynamicDimSize(kernelOutputFeatures)) {
    if (kernelOutputFeatures % batchGroupCount != 0)
      return emitOptionalError(
          location, "expects output feature dimension size (",
          kernelOutputFeatures,
          ") to be a multiple of batch_group_count. Got batch_group_count = ",
          batchGroupCount, ".");

    if (kernelOutputFeatures % featureGroupCount != 0)
      return emitOptionalError(location,
                               "expects kernel output feature dimension (",
                               kernelOutputFeatures,
                               ") to be divisible by feature_group_count. For "
                               "feature_group_count = ",
                               featureGroupCount, ".");
  }

  if (!hlo::isDynamicDimSize(inputFeatures)) {
    if (inputFeatures % featureGroupCount != 0)
      return emitOptionalError(location, "expects input feature dimension (",
                               inputFeatures,
                               ") to be a multiple of feature_group_count. Got "
                               "feature_group_count = ",
                               featureGroupCount, ".");

    if (!hlo::isDynamicDimSize(kernelInputFeatures) &&
        inputFeatures / featureGroupCount != kernelInputFeatures)
      return emitOptionalError(
          location, "expects input feature dimension (", inputFeatures,
          ") / "
          "feature_group_count = kernel input feature dimension (",
          kernelInputFeatures,
          "). Got feature_group_count = ", featureGroupCount, ".");
  }

  if (!hlo::isDynamicDimSize(inputBatch) && inputBatch % batchGroupCount != 0)
    return emitOptionalError(location, "expects input batch dimension (",
                             inputBatch,
                             ") to be divisible by "
                             "batch_group_count. Got batch_group_count = ",
                             batchGroupCount, ".");

  return success();
}

// Infer the return-shape of ConvolutionOp.
// Precondition:
//  1. Input args to ConvolutionOp 'op' are RankedTypes.
//  2. rank-of(input-type) == rank-of(output-type)
SmallVector<int64_t> inferConvolutionOpReturnShape(
    // clang-format off
    Value lhs, Value rhs,
    int64_t inputBatchDimension,
    ArrayRef<int64_t> inputSpatialDimensions,
    int64_t kernelOutputFeatureDimension,
    int64_t outputBatchDimension,
    int64_t outputFeatureDimension,
    ArrayRef<int64_t> outputSpatialDimensions,
    int64_t batchGroupCount,
    // clang-format on
    const ArrayRef<hlo::WindowDimension> window) {
  auto lhsType = lhs.getType().cast<RankedTensorType>();
  SmallVector<int64_t> outputDimensions(lhsType.getShape().size(),
                                        ShapedType::kDynamicSize);

  // Infer the output spatial dimensions.
  auto numSpatialDims = inputSpatialDimensions.size();
  SmallVector<int64_t> inputSpatialDimVals(numSpatialDims);
  for (int64_t i = 0; i < static_cast<int64_t>(numSpatialDims); ++i)
    inputSpatialDimVals[i] = lhsType.getShape()[inputSpatialDimensions[i]];

  auto windowOutputShape =
      hlo::inferWindowOutputShape(inputSpatialDimVals, window);

  for (int64_t i = 0; i < static_cast<int64_t>(window.size()); ++i)
    outputDimensions[outputSpatialDimensions[i]] = windowOutputShape[i];

  // Infer the output-batch-dimension and output-feature-dimension.
  auto rhsType = rhs.getType().cast<RankedTensorType>();
  const int64_t inputBatch = lhsType.getShape()[inputBatchDimension];
  const int64_t kernelOutputFeatures =
      rhsType.getShape()[kernelOutputFeatureDimension];

  outputDimensions[outputBatchDimension] = hlo::isDynamicDimSize(inputBatch)
                                               ? ShapedType::kDynamicSize
                                               : inputBatch / batchGroupCount;
  outputDimensions[outputFeatureDimension] = kernelOutputFeatures;

  return outputDimensions;
}

}  // namespace

/*
 * We intend to verify the following properties
 *  P1. Verify the input, kernel types.
 *  P2. Verify the convolution atributes.
 *  P3. Verify and collect the window atributes.
 *  P4. Verify the return shape.
 *      TODO(b/232574102): Verify the element-type of return-value.
 */
LogicalResult ConvolutionOp::inferReturnTypeComponents(
    MLIRContext*, Optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  ConvolutionOp::Adaptor adaptor(operands, attributes, regions);

  Value lhs = adaptor.getLhs();
  Value rhs = adaptor.getRhs();
  Optional<DenseIntElementsAttr> windowStrides = adaptor.getWindowStrides();
  Optional<DenseIntElementsAttr> padding = adaptor.getPadding();
  Optional<DenseIntElementsAttr> lhsDilation = adaptor.getLhsDilation();
  Optional<DenseIntElementsAttr> rhsDilation = adaptor.getRhsDilation();

  int64_t inputBatchDimension =
      adaptor.getDimensionNumbers().getInputBatchDimension();
  int64_t inputFeatureDimension =
      adaptor.getDimensionNumbers().getInputFeatureDimension();
  ArrayRef<int64_t> inputSpatialDimensions =
      adaptor.getDimensionNumbers().getInputSpatialDimensions();
  int64_t kernelInputFeatureDimension =
      adaptor.getDimensionNumbers().getKernelInputFeatureDimension();
  int64_t kernelOutputFeatureDimension =
      adaptor.getDimensionNumbers().getKernelOutputFeatureDimension();
  ArrayRef<int64_t> kernelSpatialDimensions =
      adaptor.getDimensionNumbers().getKernelSpatialDimensions();
  int64_t outputBatchDimension =
      adaptor.getDimensionNumbers().getOutputBatchDimension();
  int64_t outputFeatureDimension =
      adaptor.getDimensionNumbers().getOutputFeatureDimension();
  ArrayRef<int64_t> outputSpatialDimensions =
      adaptor.getDimensionNumbers().getOutputSpatialDimensions();

  int64_t featureGroupCount = adaptor.getFeatureGroupCount();
  int64_t batchGroupCount = adaptor.getBatchGroupCount();

  auto lhsType = lhs.getType().dyn_cast<RankedTensorType>();
  auto rhsType = rhs.getType().dyn_cast<RankedTensorType>();

  if (!lhsType || !rhsType) {
    inferredReturnShapes.emplace_back(lhsType.getElementType());
    return success();
  }

  // P1.
  int numDims = lhsType.getRank();
  if (numDims != rhsType.getRank())
    return emitOptionalError(location,
                             "expects convolution arguments to have same "
                             "number of dimensions. Got: ",
                             lhsType, " and ", rhsType, ".");

  if (numDims < 2)
    return emitOptionalError(
        location,
        "expects convolution arguments to have >= 2 dimensions. Got: ", lhsType,
        " and ", rhsType, ".");

  // P2.
  if (failed(verifyConvolutionAttributes(
          lhs, rhs,
          // clang-format off
          inputBatchDimension,
          inputFeatureDimension,
          inputSpatialDimensions,
          kernelInputFeatureDimension,
          kernelOutputFeatureDimension,
          kernelSpatialDimensions,
          outputBatchDimension,
          outputFeatureDimension,
          outputSpatialDimensions,
          featureGroupCount,
          batchGroupCount,
          // clang-format on
          location)))
    return failure();

  if ((size_t)numDims != inputSpatialDimensions.size() + 2)
    return emitOptionalError(location, "expects convolution arguments to have ",
                             inputSpatialDimensions.size() + 2,
                             " dimensions. Got: ", numDims);

  // P3.
  SmallVector<int64_t> windowDimensions(kernelSpatialDimensions.size());
  for (size_t i = 0; i < windowDimensions.size(); i++)
    windowDimensions[i] = rhsType.getShape()[kernelSpatialDimensions[i]];

  auto paddingOrErr = hlo::convertPaddingAttribute(padding, location);
  if (failed(paddingOrErr)) return failure();

  // TODO: add missing tests for ConvolutionOp.
  auto windowStridesOrErr =
      hlo::convert1DAttribute(windowStrides, location, "window_strides");
  if (failed(windowStridesOrErr)) return failure();
  auto lhsDilationOrErr =
      hlo::convert1DAttribute(lhsDilation, location, "lhs_dilation");
  if (failed(lhsDilationOrErr)) return failure();
  auto rhsDilationOrErr =
      hlo::convert1DAttribute(rhsDilation, location, "rhs_dilation");
  if (failed(rhsDilationOrErr)) return failure();
  auto windowOrErr = hlo::verifyWindowAttributesAndInferWindowDimensions(
      windowDimensions, *windowStridesOrErr, *paddingOrErr, *lhsDilationOrErr,
      *rhsDilationOrErr, location);
  if (failed(windowOrErr)) return failure();

  // P4.
  auto expectedReturnShape = inferConvolutionOpReturnShape(
      lhs, rhs, inputBatchDimension, inputSpatialDimensions,
      kernelOutputFeatureDimension, outputBatchDimension,
      outputFeatureDimension, outputSpatialDimensions, batchGroupCount,
      *windowOrErr);

  inferredReturnShapes.emplace_back(expectedReturnShape,
                                    lhsType.getElementType());

  return success();
}

//===----------------------------------------------------------------------===//
// ConvertOp
//===----------------------------------------------------------------------===//

void ConvertOp::build(OpBuilder& builder, OperationState& result, Value operand,
                      Type resultElementTy) {
  Type resultTy;
  Type operandTy = operand.getType();
  if (auto rankedTy = operandTy.dyn_cast<RankedTensorType>()) {
    resultTy = RankedTensorType::get(rankedTy.getShape(), resultElementTy);
  } else {
    resultTy = UnrankedTensorType::get(resultElementTy);
  }
  build(builder, result, resultTy, operand);
}

//===----------------------------------------------------------------------===//
// GetTupleElementOp
//===----------------------------------------------------------------------===//

LogicalResult GetTupleElementOp::verify() {
  auto indexVal = getIndex();
  auto operandType = getOperand().getType().cast<TupleType>();
  if (indexVal >= operandType.size()) {
    return emitOpError(
        llvm::formatv("index {0} is out of bounds of operand with size {1}",
                      indexVal, operandType.size()));
  }

  auto expectedType = operandType.getType(indexVal);
  if (getType() != expectedType) {
    return emitOpError(llvm::formatv("has return type {0}, but expected {1}",
                                     getType(), expectedType));
  }
  return success();
}

//===----------------------------------------------------------------------===//
// TupleOp
//===----------------------------------------------------------------------===//

LogicalResult TupleOp::verify() {
  auto opType = getType().dyn_cast<TupleType>();
  if (!opType) return emitOpError("tuple op with non-tuple result");
  if (getNumOperands() != opType.size())
    return emitOpError(
        "number of operands to tuple expected to match number of types in "
        "resultant tuple type");
  for (const auto& it :
       llvm::enumerate(llvm::zip_first(getOperandTypes(), opType.getTypes()))) {
    if (std::get<0>(it.value()) != std::get<1>(it.value()))
      return emitOpError("has return type mismatch at ")
             << it.index() << "th value (" << std::get<0>(it.value())
             << " != " << std::get<1>(it.value()) << ")";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// AllToAllOp
//===----------------------------------------------------------------------===//

LogicalResult AllToAllOp::inferReturnTypeComponents(
    MLIRContext*, Optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  AllToAllOp::Adaptor adaptor(operands, attributes, regions);
  Type operandType = adaptor.getOperand().getType();
  RankedTensorType operandRankedType = operandType.dyn_cast<RankedTensorType>();
  if (!operandRankedType) {
    inferredReturnShapes.emplace_back(
        operandType.cast<TensorType>().getElementType());
    return success();
  }

  int64_t inputRank = operandRankedType.getRank();
  int64_t splitDimension = static_cast<int64_t>(adaptor.getSplitDimension());
  int64_t concatDimension = static_cast<int64_t>(adaptor.getConcatDimension());
  if (splitDimension >= inputRank || splitDimension < 0) {
    return emitOptionalError(location, "AllToAll split_dimension ",
                             splitDimension,
                             " is out-of-bounds for input rank ", inputRank);
  }
  if (concatDimension >= inputRank || concatDimension < 0) {
    return emitOptionalError(location, "AllToAll concat_dimension ",
                             concatDimension,
                             " is out-of-bounds for input rank ", inputRank);
  }

  // If operand is ranked, size of split dimension should be a multiple of split
  // count.
  int64_t splitCount = adaptor.getSplitCount();
  auto splitDimSize = operandRankedType.getDimSize(splitDimension);
  if (splitDimSize % splitCount != 0) {
    return emitOptionalError(
        location, "split dimension has size ", splitDimSize,
        ", expected to be a multiple of split_count ", splitCount);
  }
  SmallVector<int64_t> resultShape(operandRankedType.getShape().begin(),
                                   operandRankedType.getShape().end());
  resultShape[splitDimension] /= splitCount;
  resultShape[concatDimension] *= splitCount;
  inferredReturnShapes.emplace_back(resultShape,
                                    operandRankedType.getElementType());
  return success();
}

//===----------------------------------------------------------------------===//
// AllGatherOp
//===----------------------------------------------------------------------===//

LogicalResult AllGatherOp::verify() {
  // If operand and result are both ranked, then the size of the gather
  // dimension in the result should be a multiple of the size of the gather
  // dimension in the operand.
  auto operandType = getOperand().getType().dyn_cast<RankedTensorType>();
  auto resultType = getType().dyn_cast<RankedTensorType>();
  uint64_t allGatherDimIndex = getAllGatherDim();
  if (!operandType || !resultType ||
      operandType.isDynamicDim(allGatherDimIndex) ||
      resultType.isDynamicDim(allGatherDimIndex))
    return success();
  if (operandType.getDimSize(allGatherDimIndex) == 0)
    return emitOpError() << "operand gather dimension cannot be zero.";
  if ((resultType.getDimSize(allGatherDimIndex) %
       operandType.getDimSize(allGatherDimIndex)) != 0)
    return emitOpError()
           << "result gather dimension has size "
           << resultType.getDimSize(allGatherDimIndex)
           << ", expected to be a multiple of operand gather dimension size "
           << operandType.getDimSize(allGatherDimIndex);

  return success();
}

//===----------------------------------------------------------------------===//
// BatchNormGradOp
//===----------------------------------------------------------------------===//

LogicalResult BatchNormGradOp::inferReturnTypeComponents(
    MLIRContext* context, Optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  BatchNormGradOp::Adaptor adaptor(operands, attributes, regions);
  return hlo::inferBatchNormGradOp(
      location, adaptor.getOperand(), adaptor.getScale(),
      adaptor.getFeatureIndex(), inferredReturnShapes);
}

//===----------------------------------------------------------------------===//
// BatchNormTrainingOp
//===----------------------------------------------------------------------===//

LogicalResult BatchNormTrainingOp::inferReturnTypeComponents(
    MLIRContext* context, Optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  BatchNormTrainingOp::Adaptor adaptor(operands, attributes, regions);
  return hlo::inferBatchNormTrainingOp(
      location, adaptor.getOperand(), adaptor.getScale(),
      adaptor.getFeatureIndex(), inferredReturnShapes);
}

//===----------------------------------------------------------------------===//
// BatchNormInferenceOp
//===----------------------------------------------------------------------===//

LogicalResult BatchNormInferenceOp::inferReturnTypeComponents(
    MLIRContext* context, Optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  BatchNormInferenceOp::Adaptor adaptor(operands, attributes, regions);
  return hlo::inferBatchNormInferenceOp(
      location, adaptor.getOperand(), adaptor.getScale(),
      adaptor.getFeatureIndex(), inferredReturnShapes);
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

/*
 * We intend to verify the following properties
 * P1. We cannot convert between complex and real types (cf xla)
 * P3. The dimensions of the operand and the target
 * shape must match, except that the shape with the smaller element bitwidth has
 * an appropriately-sized additional innermost dimension, e.g.
 * ... x f32 => [bitcast_convert] => ... x 4 x i8
 * ... x 4 x i8 => [bitcast_convert] => ... x f32
 */
LogicalResult BitcastConvertOp::verify() {
  auto operandTensorType = getOperand().getType().cast<TensorType>();
  auto targetTensorType = getResult().getType().cast<TensorType>();

  // P1.
  auto targetElt = targetTensorType.getElementType();
  auto operandElt = operandTensorType.getElementType();
  if (targetElt.isa<ComplexType>() != operandElt.isa<ComplexType>()) {
    return emitOpError()
           << "cannot convert between real and complex types, but got: "
           << operandTensorType << " and " << targetTensorType;
  }

  auto targetEltBitwidth = hlo::potentiallyComplexBitwidth(targetElt);
  auto operandEltBitwidth = hlo::potentiallyComplexBitwidth(operandElt);

  // P2.
  auto operandType = operandTensorType.dyn_cast<RankedTensorType>();
  auto targetType = targetTensorType.dyn_cast<RankedTensorType>();
  if (!operandType || !targetType) return success();

  auto targetShape = targetType.getShape();
  auto operandShape = operandType.getShape();
  ArrayRef<int64_t> smallerEltShape, biggerEltShape;
  Type smallerElt, biggerElt;
  if (operandEltBitwidth < targetEltBitwidth) {
    smallerEltShape = operandShape;
    smallerElt = operandElt;
    biggerEltShape = targetShape;
    biggerElt = targetElt;
  } else {
    smallerEltShape = targetShape;
    smallerElt = targetElt;
    biggerEltShape = operandShape;
    biggerElt = operandElt;
  }

  ArrayRef<int64_t> smallerEltPrefix;
  auto smallerEltBitwidth = std::min(targetEltBitwidth, operandEltBitwidth);
  auto biggerEltBitwidth = std::max(targetEltBitwidth, operandEltBitwidth);
  if (operandEltBitwidth != targetEltBitwidth) {
    if (smallerEltShape.empty()) {
      return emitOpError() << "does not allow the smaller element type to be "
                              "part of a 0d tensor, but got: "
                           << operandType << " and " << targetType << ".";
    }
    smallerEltPrefix = smallerEltShape.drop_back();
    if (!hlo::isDynamicDimSize(smallerEltShape.back()) &&
        smallerEltShape.back() * smallerEltBitwidth != biggerEltBitwidth) {
      return emitOpError() << "requires compatible bitwidths. "
                           << "Got: " << operandType << " and " << targetType
                           << ", but " << smallerEltBitwidth << " * "
                           << smallerEltShape.back()
                           << " != " << biggerEltBitwidth << ".";
    }
  } else {
    smallerEltPrefix = smallerEltShape;
  }

  for (auto it : llvm::zip(smallerEltPrefix, biggerEltShape)) {
    auto targetDim = std::get<0>(it);
    auto operandDim = std::get<1>(it);
    if (!hlo::isDynamicDimSize(targetDim) &&
        !hlo::isDynamicDimSize(operandDim)) {
      if (targetDim != operandDim) {
        return emitOpError() << "operand and result shapes must match except "
                                "for the innermost dimension of the shape with "
                                "the smaller element type. Got: "
                             << operandType << " and " << targetType << ".";
      }
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// BroadcastOp
//===----------------------------------------------------------------------===//

// TODO(b/129012527) These should be expressed as type constraints.
LogicalResult BroadcastOp::verify() {
  auto sizes = getBroadcastSizes();
  auto sizesType = sizes.getType();
  auto sizesRank = sizesType.getRank();
  if (sizesRank != 1) {
    return emitOpError(llvm::formatv(
        "broadcast_sizes has rank {0} instead of rank 1", sizesRank));
  }

  return success();
}

LogicalResult BroadcastOp::inferReturnTypeComponents(
    MLIRContext*, Optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  BroadcastOp::Adaptor adaptor(operands, attributes, regions);
  Value operand = adaptor.getOperand();
  auto operandType = operand.getType().dyn_cast<RankedTensorType>();
  if (!operandType) return failure();

  Type elementTy = operandType.getElementType();
  auto dimensionAttr = adaptor.getBroadcastSizes();
  for (int64_t size : dimensionAttr.getValues<int64_t>()) {
    if (size < 0)
      return emitOptionalError(location,
                               "Broadcast with negative dimension size ", size);
  }
  SmallVector<int64_t> shapeValues(dimensionAttr.getValues<int64_t>());
  llvm::append_range(shapeValues, operandType.getShape());

  inferredReturnShapes.emplace_back(shapeValues, elementTy);
  return success();
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
  for (const auto& size : getBroadcastSizes()) {
    shapeValues.push_back(
        builder.create<arith::ConstantIndexOp>(loc, size.getZExtValue()));
  }

  // Collect the operand sizes.
  for (auto index : llvm::seq<int64_t>(0, operandType.getRank())) {
    shapeValues.push_back(
        builder.createOrFold<tensor::DimOp>(loc, operand, index));
  }

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
  auto operandType = getOperand().getType().dyn_cast<RankedTensorType>();
  if (!operandType) {
    // The following verification checks all depend on knowing the rank of
    // the operand. Bail out now if we don't know the rank of the operand.
    return success();
  }

  auto operandRank = operandType.getRank();
  if (!getBroadcastDimensions()) {
    if (operandRank == 0) {
      return success();
    }
    return emitOpError(
        llvm::formatv("broadcast_dimensions is absent, but required because "
                      "operand has non-zero rank ({0})",
                      operandRank));
  }

  auto dimensionsType = getBroadcastDimensions().getType();
  auto dimensionsRank = dimensionsType.getRank();
  if (dimensionsRank != 1) {
    return emitOpError(llvm::formatv(
        "broadcast_dimensions has rank {0} instead of rank 1", dimensionsRank));
  }

  auto dimensionsSize = dimensionsType.getNumElements();
  if (dimensionsSize != operandRank) {
    return emitOpError(llvm::formatv(
        "broadcast_dimensions size ({0}) does not match operand rank ({1})",
        dimensionsSize, operandRank));
  }

  auto dimensions =
      llvm::to_vector(getBroadcastDimensions().getValues<int64_t>());
  if (hasDuplicates(dimensions))
    return emitOpError("broadcast_dimensions should not have duplicates");

  auto resultType = getResult().getType().cast<RankedTensorType>();
  auto resultRank = resultType.getRank();
  for (int i = 0; i != dimensionsSize; ++i) {
    auto dimIndex = dimensions[i];
    if (dimIndex >= resultRank) {
      return emitOpError(
          llvm::formatv("broadcast_dimensions contains invalid value {0} for "
                        "result with rank {1}",
                        dimIndex, resultRank));
    }

    if (!operandType.isDynamicDim(i)) {
      auto dimSize = operandType.getDimSize(i);
      auto resultDimSize = resultType.getDimSize(dimIndex);
      if (dimSize != 1 && dimSize != resultDimSize) {
        return emitOpError(
            llvm::formatv("size of operand dimension {0} ({1}) is not equal to "
                          "1 or size of result dimension {2} ({3})",
                          i, dimSize, dimIndex, resultDimSize));
      }
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// DynamicBroadcastInDimOp
//===----------------------------------------------------------------------===//

LogicalResult DynamicBroadcastInDimOp::verify() {
  auto operandType = getOperand().getType().dyn_cast<RankedTensorType>();
  auto resultType = getResult().getType().dyn_cast<RankedTensorType>();

  // If either the operand or result are unranked, there is very little
  // to verify statically.
  if (!operandType || !resultType) {
    return success();
  }

  auto outputDimensionsType =
      getOutputDimensions().getType().cast<RankedTensorType>();
  auto outputDimensionsSize = outputDimensionsType.getDimSize(0);
  auto operandRank = operandType.getRank();
  auto resultRank = resultType.getRank();

  // Verify broadcast_dimensions.
  auto bcastDimensions = getBroadcastDimensions();
  auto bcastDimensionsType = getBroadcastDimensions().getType();
  auto bcastDimensionsRank = bcastDimensionsType.getRank();
  // TODO(laurenzo): Update the BroadcastDimAttr to constrain its rank to 1.
  if (bcastDimensionsRank != 1) {
    return emitOpError(
        llvm::formatv("broadcast_dimensions has rank {0} instead of rank 1",
                      bcastDimensionsRank));
  }

  auto bcastDimensionsSize = bcastDimensionsType.getNumElements();
  if (bcastDimensionsSize != operandRank) {
    return emitOpError(llvm::formatv(
        "broadcast_dimensions size ({0}) does not match operand rank ({1})",
        bcastDimensionsSize, operandRank));
  }

  if (resultRank < operandRank) {
    return emitOpError(
        llvm::formatv("result rank ({0}) is less than operand rank ({1})",
                      resultRank, operandRank));
  }

  for (int i = 0; i != bcastDimensionsSize; ++i) {
    auto dimIndex = bcastDimensions.getValues<int64_t>()[i];
    if (dimIndex >= resultRank) {
      return emitOpError(
          llvm::formatv("broadcast_dimensions contains invalid value {0} for "
                        "result with rank {1}",
                        dimIndex, resultRank));
    }

    auto dimSize = operandType.getDimSize(i);
    auto resultDimSize = resultType.getDimSize(dimIndex);
    // Note: verifyCompatibleShapes doesn't consider size-1 broadcasting, so we
    // add a manual check for this.
    if (dimSize != 1 && failed(verifyCompatibleShape(dimSize, resultDimSize))) {
      return emitOpError(
          llvm::formatv("size of operand dimension {0} ({1}) is not compatible "
                        "with size of result dimension {2} ({3})",
                        i, dimSize, dimIndex, resultDimSize));
    }
  }

  if (outputDimensionsSize != resultRank) {
    return emitOpError(
        llvm::formatv("result rank ({0}) is not equal to number of output "
                      "dimensions ({1})",
                      resultRank, outputDimensionsSize));
  }

  // Verify that the known expanding and non-expanding dimensions are a subset
  // of the operand's dimensions.
  int64_t numKnownExpansionBehavior = 0;
  DenseSet<int64_t> knownExpansionBehavior;
  auto collectExpansionBehaviorDims =
      [&](const Optional<DenseIntElementsAttr>& attr) {
        if (!attr) return;
        for (const APInt& it : *attr) {
          numKnownExpansionBehavior++;
          knownExpansionBehavior.insert(it.getLimitedValue());
        }
      };
  collectExpansionBehaviorDims(getKnownExpandingDimensions());
  collectExpansionBehaviorDims(getKnownNonexpandingDimensions());
  if (knownExpansionBehavior.size() != numKnownExpansionBehavior) {
    return emitOpError(
        "duplicate expansion hint for at least one operand dimension");
  }
  for (int64_t i : knownExpansionBehavior) {
    if (i < 0 || i >= operandRank) {
      return emitOpError(
          llvm::formatv("hint for expanding dimension {0} does not refer to a "
                        "valid operand dimension",
                        i));
    }
  }

  return success();
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

LogicalResult ClampOp::verify() {
  auto operandType = getOperand().getType().cast<RankedTensorType>();
  auto operandShape = operandType.getShape();
  auto minType = getMin().getType().cast<RankedTensorType>();

  auto minShape = minType.getShape();
  if (failed(verifyCompatibleShape(minType, operandType)) &&
      minType.getRank() != 0) {
    return emitOpError(llvm::formatv(
        "min shape [{0}] is not scalar and is not compatible to operand shape "
        "[{1}]",
        llvm::make_range(minShape.begin(), minShape.end()),
        llvm::make_range(operandShape.begin(), operandShape.end())));
  }

  auto maxType = getMax().getType().cast<RankedTensorType>();
  auto maxShape = maxType.getShape();
  if (failed(verifyCompatibleShape(maxType, operandType)) &&
      maxType.getRank() != 0) {
    return emitOpError(llvm::formatv(
        "max shape [{0}] is not scalar and is not compatible to operand shape "
        "[{1}]",
        llvm::make_range(maxShape.begin(), maxShape.end()),
        llvm::make_range(operandShape.begin(), operandShape.end())));
  }

  return success();
}

LogicalResult ClampOp::inferReturnTypeComponents(
    MLIRContext*, Optional<Location> /*location*/, ValueShapeRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  ClampOp::Adaptor adaptor(operands, attributes, regions);
  RankedTensorType operandType =
      adaptor.getOperand().getType().cast<RankedTensorType>();
  inferredReturnShapes.emplace_back(operandType.cast<ShapedType>());
  return success();
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
    MLIRContext*, Optional<Location>, ValueRange operands, DictionaryAttr,
    RegionRange, SmallVectorImpl<Type>& inferredReturnTypes) {
  TensorType operandType = operands[0].getType().cast<TensorType>();
  ComplexType elementTy = ComplexType::get(operandType.getElementType());
  inferredReturnTypes.push_back(
      hlo::getSameShapeTensorType(operandType, elementTy));
  return success();
}

//===----------------------------------------------------------------------===//
// ImagOp
//===----------------------------------------------------------------------===//

LogicalResult ImagOp::inferReturnTypes(
    MLIRContext*, Optional<Location>, ValueRange operands, DictionaryAttr,
    RegionRange, SmallVectorImpl<Type>& inferredReturnTypes) {
  inferredReturnTypes.push_back(
      hlo::createRealType(operands[0].getType().cast<TensorType>()));
  return success();
}

//===----------------------------------------------------------------------===//
// IsFiniteOp
//===----------------------------------------------------------------------===//

LogicalResult IsFiniteOp::inferReturnTypes(
    MLIRContext* ctx, Optional<Location>, ValueRange operands, DictionaryAttr,
    RegionRange, SmallVectorImpl<Type>& inferredReturnTypes) {
  auto argTy = operands.front().getType().cast<TensorType>();
  Builder b(ctx);
  inferredReturnTypes.push_back(
      hlo::getSameShapeTensorType(argTy, b.getI1Type()));
  return success();
}

//===----------------------------------------------------------------------===//
// RealOp
//===----------------------------------------------------------------------===//

LogicalResult RealOp::inferReturnTypes(
    MLIRContext*, Optional<Location>, ValueRange operands, DictionaryAttr,
    RegionRange, SmallVectorImpl<Type>& inferredReturnTypes) {
  inferredReturnTypes.push_back(
      hlo::createRealType(operands[0].getType().cast<TensorType>()));
  return success();
}

//===----------------------------------------------------------------------===//
// ConcatenateOp
//===----------------------------------------------------------------------===//

LogicalResult ConcatenateOp::inferReturnTypes(
    MLIRContext*, Optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  ConcatenateOp::Adaptor adaptor(operands, attributes, regions);
  return hlo::inferConcatenateOp(location, adaptor.getInputs(),
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
  auto resultType = getResult().getType().dyn_cast<RankedTensorType>();
  auto outputShapeType =
      getOutputShape().getType().dyn_cast<RankedTensorType>();
  if (resultType && outputShapeType && outputShapeType.hasStaticShape() &&
      outputShapeType.getDimSize(0) != resultType.getRank()) {
    return emitError() << "output should have a rank equal to the number of "
                          "elements in output_shape";
  }
  return success();
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

// Verifies that the number of slice sizes and the number of start indices match
LogicalResult DynamicSliceOp::verify() {
  int numSliceSizes = getSliceSizes().getNumElements();
  int numStartIndices = getStartIndices().size();
  if (numStartIndices != numSliceSizes) {
    return emitOpError() << "has mismatched number of slice sizes ("
                         << numSliceSizes << ") and number of start indices ("
                         << numStartIndices << ")";
  }
  auto operandType = getOperand().getType().dyn_cast<RankedTensorType>();
  if (!operandType) return failure();

  if (operandType.getRank() != numStartIndices) {
    return emitOpError() << "has mismatched number of start indices ("
                         << numStartIndices << ") and the rank of operand ("
                         << operandType.getRank() << ")";
  }

  for (int i = 0; i < numSliceSizes; ++i) {
    int64_t sliceSize = getSliceSizes().getValues<int64_t>()[i];
    if (sliceSize < 0) {
      return emitOpError() << "has negative size index to dynamic slice: "
                           << sliceSize;
    }
    if (!operandType.isDynamicDim(i)) {
      int64_t dimSize = operandType.getDimSize(i);
      if (sliceSize > dimSize) {
        return emitOpError() << "has slice size " << sliceSize
                             << " greater than dimension size " << dimSize
                             << " in dimension " << i << " of operand";
      }
    }
  }
  return success();
}

LogicalResult DynamicSliceOp::inferReturnTypeComponents(
    MLIRContext*, Optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  DynamicSliceOp::Adaptor adaptor(operands, attributes, regions);
  Value operand = adaptor.getOperand();
  auto operandType = operand.getType().dyn_cast<RankedTensorType>();
  if (!operandType) return failure();

  auto sliceSizes = adaptor.getSliceSizes();
  Type elementTy = operandType.getElementType();
  inferredReturnShapes.emplace_back(sliceSizes.getValues<int64_t>(), elementTy);
  return success();
}

//===----------------------------------------------------------------------===//
// RealDynamicSliceOp
//===----------------------------------------------------------------------===//
// Verifies that operand rank matches start_indices/limit_indices/strides size
LogicalResult RealDynamicSliceOp::verify() {
  auto inputType = getOperand().getType().dyn_cast<RankedTensorType>();
  // If operand is unranked, there is very little to verify statically.
  if (!inputType) return success();
  int inputRank = inputType.getRank();

  auto startType = getStartIndices().getType().cast<RankedTensorType>();
  auto limitType = getLimitIndices().getType().cast<RankedTensorType>();
  auto stridesType = getStrides().getType().cast<RankedTensorType>();

  if (inputRank != startType.getNumElements()) {
    return emitOpError() << "has mismatched number of operand rank ("
                         << inputRank << ") and start_indices size ("
                         << startType.getNumElements() << ")";
  }

  if (inputRank != limitType.getNumElements()) {
    return emitOpError() << "has mismatched number of operand rank ("
                         << inputRank << ") and limit_indices size ("
                         << limitType.getNumElements() << ")";
  }

  if (inputRank != stridesType.getNumElements()) {
    return emitOpError() << "has mismatched number of operand rank ("
                         << inputRank << ") and strides size ("
                         << stridesType.getNumElements() << ")";
  }

  return success();
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

// Checks that the result type is of the form `zero_or_more_type(s),
// stablehlo::token`
LogicalResult InfeedOp::verify() {
  auto resultTypes = getResultTypes();
  if (resultTypes.empty())
    return emitOpError()
           << "result is expected to be at least of size 1, but got "
           << resultTypes.size();

  if (!resultTypes[resultTypes.size() - 1].isa<TokenType>())
    return emitOpError() << "last element of result types is expected to "
                            "be of token type, but got "
                         << resultTypes[resultTypes.size() - 1];

  // Verify layout attribute
  constexpr char kLayoutAttr[] = "layout";
  if (!getOperation()->hasAttr(kLayoutAttr)) return success();

  mlir::ArrayAttr layout =
      getOperation()->getAttrOfType<mlir::ArrayAttr>(kLayoutAttr);
  if (!layout)
    return emitOpError() << "layout-attribute expected to be of array-type.";

  if (layout.size() != resultTypes.size() - 1) {
    return emitOpError() << "layout-attribute size must be "
                         << resultTypes.size() - 1
                         << " (which is the number of "
                            "op-results - 1 (for token result)), but got "
                         << layout.size();
  }

  for (auto childLayout : layout) {
    mlir::ArrayAttr childLayoutArr = childLayout.dyn_cast<mlir::ArrayAttr>();
    if (!childLayoutArr) {
      return emitOpError() << "layout-attribute expected to have "
                              "elements of type array, but got "
                           << childLayout;
    }

    for (auto i : childLayoutArr) {
      mlir::IntegerAttr attr = i.dyn_cast<mlir::IntegerAttr>();
      if (!attr) {
        return emitOpError() << "layout-attribute's leaf elements are "
                                "expected to be of type integer, but got "
                             << i;
      }
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// MapOp
//===----------------------------------------------------------------------===//

LogicalResult MapOp::inferReturnTypeComponents(
    MLIRContext*, Optional<Location> location, ValueShapeRange operands,
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
// RecvOp
//===----------------------------------------------------------------------===//

// Checks that the result type is of the form `zero_or_more_type(s),
// stablehlo::token`
LogicalResult RecvOp::verify() {
  auto resultTypes = getResultTypes();
  if (resultTypes.empty())
    return emitOpError()
           << "result is expected to be at least of size 1, but got "
           << resultTypes.size();
  if (!resultTypes[resultTypes.size() - 1].isa<TokenType>())
    return emitOpError() << "last element of result types is expected to "
                            "be of token type, but got "
                         << resultTypes[resultTypes.size() - 1];
  return success();
}

//===----------------------------------------------------------------------===//
// ReduceWindowOp
//===----------------------------------------------------------------------===//

LogicalResult ReduceWindowOp::inferReturnTypeComponents(
    MLIRContext*, Optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  ReduceWindowOp::Adaptor adaptor(operands, attributes, regions);
  return hlo::inferReduceWindowOp(
      location, adaptor.getInputs(), adaptor.getInitValues(),
      adaptor.getWindowDimensions(), adaptor.getWindowStrides(),
      adaptor.getBaseDilations(), adaptor.getWindowDilations(),
      adaptor.getPadding(), adaptor.getBody(), inferredReturnShapes);
}

// Get the operation used for reduction applied to `result_index`th result. Its
// expected to be a binary operation that consumes `result_index`th and
// `result_index + getInputs().size`th arguments of the body.
Operation* ReduceWindowOp::getReductionOp(int resultIndex) {
  auto returnOp = cast<ReturnOp>(getBody().front().getTerminator());
  Operation* computeOp = returnOp.getResults()[resultIndex].getDefiningOp();
  if (computeOp->getNumOperands() != 2) return nullptr;
  auto arg0 = computeOp->getOperand(0).dyn_cast<BlockArgument>();
  auto arg1 = computeOp->getOperand(1).dyn_cast<BlockArgument>();
  if (!arg0 || !arg1) return nullptr;
  int64_t arg0Num = arg0.getArgNumber();
  int64_t arg1Num = arg1.getArgNumber();
  int64_t otherArgIndex = resultIndex + getInputs().size();
  if (arg0Num == resultIndex && arg1Num == otherArgIndex) return computeOp;
  if (arg0Num == otherArgIndex && arg1Num == resultIndex &&
      computeOp->hasTrait<mlir::OpTrait::IsCommutative>())
    return computeOp;
  return nullptr;
}

//===----------------------------------------------------------------------===//
// ReducePrecisionOp
//===----------------------------------------------------------------------===//

// The following property is already enforced by the ODS:
//  P0. operand element type is float
//  P1. mantissa_bits >= 0
// We intend to verify the following properties
//  P2. exponent_bits >= 1
LogicalResult ReducePrecisionOp::verify() {
  if (getExponentBits() < 1) {
    return emitOpError() << "exponent_bits must be at least 1.";
  }
  return success();
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
    SmallVector<Optional<Location>, 2> reducerLocs;
    SmallVector<Optional<Location>, 2> reducerInitLocs;
    auto parseBlockOperand =
        [&](SmallVectorImpl<OpAsmParser::UnresolvedOperand>& operands,
            SmallVectorImpl<Type>& types,
            SmallVectorImpl<Optional<Location>>& locs) -> ParseResult {
      OpAsmParser::UnresolvedOperand operand;
      Type type;
      Optional<Location> loc;
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
    Optional<Location> trailingLoc;
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

  Optional<Location> explicitLoc;
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
    MLIRContext*, Optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  ReduceOp::Adaptor adaptor(operands, attributes, regions);
  return hlo::inferReduceOp(location, adaptor.getInputs(),
                            adaptor.getInitValues(), adaptor.getDimensions(),
                            adaptor.getBody(), inferredReturnShapes);
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
    if (it != dimensions.end()) {
      continue;
    }
    Value valueDim = toShapeScalarType(
        builder.create<tensor::DimOp>(loc, inputs[0], element.index()));
    shapeValues.push_back(valueDim);
  }

  Value outputShape = builder.create<tensor::FromElementsOp>(
      loc,
      RankedTensorType::get({static_cast<int64_t>(shapeValues.size())},
                            shapeScalarType),
      shapeValues);
  for (size_t i = 0; i < inputs.size(); ++i) {
    reifiedReturnShapes.push_back(outputShape);
  }

  return success();
}

//===----------------------------------------------------------------------===//
// RngBitGeneratorOp
//===----------------------------------------------------------------------===//

// Verify that input state has the same shape as output shape
LogicalResult RngBitGeneratorOp::verify() {
  auto initialShape = getInitialState().getType().dyn_cast<RankedTensorType>();
  auto outputShape = getOutputState().getType().dyn_cast<RankedTensorType>();
  if (initialShape.getShape() != outputShape.getShape())
    return emitOpError()
           << "output state shape must match initial state shape. Got: "
           << initialShape << " and " << outputShape;
  return success();
}

//===----------------------------------------------------------------------===//
// RngOp
//===----------------------------------------------------------------------===//

LogicalResult RngOp::verify() {
  auto dist = getRngDistribution();
  if (dist == RngDistribution::UNIFORM) {
    return success();
  }
  auto muTy = getA().getType().cast<TensorType>().getElementType();
  auto sigmaTy = getB().getType().cast<TensorType>().getElementType();
  if (muTy.isa<FloatType>() && sigmaTy.isa<FloatType>()) {
    return success();
  }
  return emitOpError() << "mu and sigma must be floats";
}

LogicalResult RngOp::inferReturnTypeComponents(
    MLIRContext* context, Optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  return rngInferReturnTypeComponents(context, location, operands, attributes,
                                      regions, inferredReturnShapes);
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

LogicalResult SelectOp::verify() {
  // The operands 'on_true' and 'on_false' should have compatible types, i.e.,
  //   (a) have the same element type, and
  //   (b) have compatible shapes (i.e. the same shape and/or at least one
  //       dynamic shape)
  if (!hlo::compatibleShapeAndElementType(getOnTrue().getType(),
                                          getOnFalse().getType()))
    return emitOpError()
           << "requires compatible types for non-predicate operands";

  // The predicate, if not-scalar, should have the same shape as the remaining
  // operands.
  auto predTy = getPred().getType().dyn_cast<RankedTensorType>();
  bool predMayBeScalar = !predTy || predTy.getRank() == 0;
  if (predMayBeScalar) return success();

  if (failed(verifyCompatibleShape(getPred().getType(), getOnTrue().getType())))
    return emitOpError() << "requires the same shape for all operands";

  return success();
}

// Makes it such that a SelectOp that is a non-root operation in a DRR infers
// the return type based on operand type.
LogicalResult SelectOp::inferReturnTypeComponents(
    MLIRContext*, Optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, RegionRange,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  SelectOp::Adaptor op(operands, attributes);
  auto trueType = op.getOnTrue().getType().cast<TensorType>();
  auto falseType = op.getOnFalse().getType().cast<TensorType>();

  // The output shape should be the most general of the operand shapes at each
  // dimension.
  ShapedTypeComponents& outputType = inferredReturnShapes.emplace_back();
  if (trueType == falseType || !trueType.hasRank()) {
    outputType = ShapedTypeComponents(trueType.cast<ShapedType>());
  } else if (!falseType.hasRank()) {
    outputType = ShapedTypeComponents(falseType.cast<ShapedType>());
  } else {
    assert(trueType.getRank() == falseType.getRank());
    llvm::SmallVector<int64_t, 4> dims;
    dims.reserve(trueType.getRank());
    for (auto dim : llvm::zip(trueType.getShape(), falseType.getShape())) {
      dims.push_back(std::get<0>(dim) == std::get<1>(dim)
                         ? std::get<0>(dim)
                         : ShapedType::kDynamicSize);
    }
    outputType = ShapedTypeComponents(dims, trueType.getElementType());
  }
  return success();
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

LogicalResult SetDimensionSizeOp::verify() {
  if (auto size = this->getSize().getType().dyn_cast<RankedTensorType>()) {
    if (size.getRank() != 0)
      return emitOpError() << "size operand should be of rank-0";
  }

  return verifyDimAttr(*this);
}

// TODO(b/238903565): Switch to inferReturnTypeComponents after adding support
// for the encoding upstream.
LogicalResult SetDimensionSizeOp::inferReturnTypes(
    MLIRContext* context, Optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  Location loc = location.value_or(UnknownLoc::get(context));

  SetDimensionSizeOp::Adaptor adaptor(operands, attributes, regions);
  if (failed(adaptor.verify(loc))) return failure();

  auto inputType = adaptor.getOperand().getType().dyn_cast<RankedTensorType>();
  if (!inputType) {
    inferredReturnTypes.push_back(adaptor.getOperand().getType());
    return success();
  }

  int64_t dim = adaptor.getDimension();
  int64_t rank = inputType.getRank();
  if (dim < 0 || dim >= rank) {
    return mlir::emitError(loc) << "expects dimension to be in range [0, "
                                << rank << "); got: [" << dim << "].";
  }

  auto shape = llvm::to_vector<4>(inputType.getShape());
  llvm::SmallVector<int64_t, 4> bounds(rank, ShapedType::kDynamicSize);
  if (auto encoding =
          inputType.getEncoding().dyn_cast_or_null<TypeExtensionsAttr>())
    bounds = llvm::to_vector<4>(encoding.getBounds());

  if (shape[dim] != ShapedType::kDynamicSize) bounds[dim] = shape[dim];
  shape[dim] = ShapedType::kDynamicSize;

  DenseIntElementsAttr sizeAttr;
  if (matchPattern(adaptor.getSize(), m_Constant(&sizeAttr))) {
    int64_t splat =
        sizeAttr.getSplatValue<IntegerAttr>().getValue().getSExtValue();
    if (splat == bounds[dim]) {
      shape[dim] = splat;
      bounds[dim] = ShapedType::kDynamicSize;
    }
  }

  auto extensions = TypeExtensionsAttr::get(context, bounds);
  auto resultType =
      llvm::all_of(bounds,
                   [](int64_t v) { return v == ShapedType::kDynamicSize; })
          ? RankedTensorType::get(shape, inputType.getElementType())
          : RankedTensorType::get(shape, inputType.getElementType(),
                                  extensions);
  inferredReturnTypes.push_back(resultType);
  return success();
}

//===----------------------------------------------------------------------===//
// PadOp
//===----------------------------------------------------------------------===//

LogicalResult PadOp::inferReturnTypes(
    MLIRContext*, Optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  PadOp::Adaptor adaptor(operands, attributes, regions);
  return hlo::inferPadOp(location, adaptor.getOperand(),
                         adaptor.getPaddingValue(), adaptor.getEdgePaddingLow(),
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
  auto inputType = getOperand().getType().dyn_cast<RankedTensorType>();
  // If operand is unranked, there is very little to verify statically.
  if (!inputType) return success();
  int inputRank = inputType.getRank();

  auto padType = getPaddingValue().getType().cast<RankedTensorType>();
  if (padType.getRank() != 0) {
    return emitOpError() << "padding value type should be a rank-0";
  }

  auto paddingLowType = getEdgePaddingLow().getType().cast<RankedTensorType>();
  if (paddingLowType.getNumElements() != inputRank) {
    return emitOpError() << "edge_padding_low length("
                         << paddingLowType.getNumElements()
                         << ") must match operand rank(" << inputRank << ").";
  }

  auto paddingHighType =
      getEdgePaddingHigh().getType().cast<RankedTensorType>();
  if (paddingHighType.getNumElements() != inputRank) {
    return emitOpError() << "edge_padding_high length("
                         << paddingHighType.getNumElements()
                         << ") must match operand rank(" << inputRank << ").";
  }

  auto interiorPaddingType =
      getInteriorPadding().getType().cast<RankedTensorType>();
  if (interiorPaddingType.getNumElements() != inputRank) {
    return emitOpError() << "edge_padding_interior length("
                         << interiorPaddingType.getNumElements()
                         << ") must match operand rank(" << inputRank << ").";
  }

  auto outputType = getResult().getType().dyn_cast<RankedTensorType>();
  // If result is unranked, there is very little to verify statically.
  if (!outputType) return success();
  int outputRank = outputType.getRank();
  if (inputRank != outputRank) {
    return emitOpError() << "operand rank(" << inputRank
                         << ") must match result(" << outputRank << ").";
  }

  return success();
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
  // If the operand type is dynamically shaped there is nothing to verify.
  auto operandTy = getOperand().getType().dyn_cast<RankedTensorType>();
  if (!operandTy || !operandTy.hasStaticShape()) return success();

  // If the operand type is statically shaped (not required) the number of
  // elements must match that of the result type.
  auto resultTy = getType().cast<RankedTensorType>();
  assert(resultTy && resultTy.hasStaticShape() &&
         "result type must be statically shaped");
  int64_t numResultElements = resultTy.getNumElements();
  int64_t numOperandElements = operandTy.getNumElements();
  if (numResultElements != numOperandElements)
    return emitOpError() << "number of output elements (" << numResultElements
                         << ") doesn't match expected number of elements ("
                         << numOperandElements << ")";

  return success();
}

//===----------------------------------------------------------------------===//
// ReplicaId Op
//===----------------------------------------------------------------------===//

LogicalResult ReplicaIdOp::inferReturnTypes(
    MLIRContext* context, Optional<Location>, ValueRange operands,
    DictionaryAttr, RegionRange, SmallVectorImpl<Type>& inferredReturnTypes) {
  inferredReturnTypes.push_back(RankedTensorType::get(
      /*shape=*/{}, IntegerType::get(context, 32, IntegerType::Unsigned)));
  return success();
}

//===----------------------------------------------------------------------===//
// If Op
//===----------------------------------------------------------------------===//

LogicalResult IfOp::inferReturnTypes(
    MLIRContext*, Optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  IfOp::Adaptor adaptor(operands, attributes, regions);
  return hlo::inferIfOp(location, adaptor.getRegions(), inferredReturnTypes);
}

//===----------------------------------------------------------------------===//
// Case Op
//===----------------------------------------------------------------------===//

LogicalResult CaseOp::inferReturnTypes(
    MLIRContext*, Optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  CaseOp::Adaptor adaptor(operands, attributes, regions);
  return hlo::inferCaseOp(location, adaptor.getRegions(), inferredReturnTypes);
}

//===----------------------------------------------------------------------===//
// SliceOp
//===----------------------------------------------------------------------===//

LogicalResult SliceOp::inferReturnTypes(
    MLIRContext* context, Optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  SliceOpAdaptor adaptor(operands, attributes);
  return hlo::inferSliceOp(location, adaptor.getOperand(),
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
    MLIRContext*, Optional<Location> location, ValueShapeRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  SortOp::Adaptor adaptor(operands, attributes, regions);
  return hlo::inferSortOp(location, adaptor.getInputs(), adaptor.getDimension(),
                          adaptor.getComparator(), inferredReturnShapes);
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
    MLIRContext*, Optional<Location> loc, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  TransposeOp::Adaptor adaptor(operands, attributes, regions);
  LogicalResult result = hlo::inferTransposeOp(
      loc, adaptor.getOperand(), adaptor.getPermutation(), inferredReturnTypes);
  return result;
}

//===----------------------------------------------------------------------===//
// TriangularSolveOp
//===----------------------------------------------------------------------===//

LogicalResult TriangularSolveOp::inferReturnTypeComponents(
    MLIRContext*, Optional<Location> location, ValueShapeRange operands,
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
    MLIRContext*, Optional<Location>, ValueRange operands,
    DictionaryAttr attributes, RegionRange,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  auto tupleType = operands[0].getType().dyn_cast<TupleType>();
  if (!tupleType) return failure();

  auto indexAttr = attributes.get("index").cast<IntegerAttr>();
  auto index = indexAttr.getInt();
  if (index < 0 || index >= static_cast<int64_t>(tupleType.size()))
    return failure();

  inferredReturnTypes.push_back(tupleType.getType(index));
  return success();
}

//===----------------------------------------------------------------------===//
// TupleOp
//===----------------------------------------------------------------------===//

LogicalResult TupleOp::inferReturnTypes(
    MLIRContext* context, Optional<Location>, ValueRange operands,
    DictionaryAttr attributes, RegionRange,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  inferredReturnTypes.push_back(TupleType::get(context, TypeRange(operands)));
  return success();
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
    mlir::MLIRContext* ctx, llvm::Optional<mlir::Location>,
    ValueShapeRange operands, mlir::DictionaryAttr, mlir::RegionRange,
    llvm::SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnTypes) {
  ShapedTypeComponents& components =
      inferredReturnTypes.emplace_back(IntegerType::get(ctx, /*width=*/1));
  auto argTy = operands.front().getType().cast<TensorType>();
  if (argTy.hasRank()) {
    components =
        ShapedTypeComponents(argTy.getShape(), components.getElementType());
  }
  return success();
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

namespace {
// Infer the return-type of SelectAndScatterOp.
TensorType inferSelectAndScatterOpReturnType(
    TensorType operandType, const ArrayRef<hlo::WindowDimension> window) {
  if (!operandType.hasRank())
    return UnrankedTensorType::get(operandType.getElementType());

  return RankedTensorType::get(
      hlo::inferWindowOutputShape(operandType.getShape(), window),
      operandType.getElementType());
}
}  // namespace

//  We intend to verify the following properties:
//   P1. Check if the select function has a proper shape of (T,T) -> PRED, where
//        T is a 0-D tensor with element-type same as 'operand' element-type.
//   P2. Verify scatter-computation type.
//   P3. size-of(window_dimension) == rank-of(input),
//         where input is an element of 'inputs'.
//   P4. Verify and collect the window attributes.
//   P5. Verify the return type matches the operand-type.
//   P6. Check if the result type of window operation matches the source type.
LogicalResult SelectAndScatterOp::verify() {
  auto operandType = getOperand().getType().cast<TensorType>();
  auto initValueType = getInitValue().getType().cast<TensorType>();
  auto sourceType = getSource().getType().cast<TensorType>();
  auto resultType = getResult().getType().cast<TensorType>();

  // P1.
  Block& selectBlock = getSelect().front();

  if (selectBlock.getArguments().size() != 2)
    return emitOpError()
           << "expects the select-region to take 2 parameters, but takes "
           << selectBlock.getArguments().size();

  Type expectedSelectArgType =
      RankedTensorType::get({}, operandType.getElementType());
  for (const auto& selectArgIt : llvm::enumerate(selectBlock.getArguments()))
    if (!hlo::compatibleShapeAndElementType(expectedSelectArgType,
                                            selectArgIt.value().getType(),
                                            /*ignoreFpPrecision=*/true))
      return emitOpError()
             << "expects the type of select-region's parameter at index "
             << selectArgIt.index() << " to be " << expectedSelectArgType
             << ", but got " << selectArgIt.value().getType();

  auto selectResult = selectBlock.getTerminator()->getOperands();
  if (selectResult.size() != 1)
    return emitOpError()
           << "expects select-region to return single value, but got: "
           << selectResult.size();

  auto selectResultType = selectResult[0].getType().dyn_cast<TensorType>();
  if (!selectResultType || !selectResultType.getElementType().isInteger(1) ||
      (selectResultType.hasRank() &&
       selectResultType.cast<RankedTensorType>().getRank() != 0))
    return emitOpError() << "expects the return-type of select-region to be "
                            "tensor<i1>, but got: "
                         << selectResult[0].getType();

  // P2.
  Block& scatterBlock = getScatter().front();
  SmallVector<TensorType> accumulatorSubshapes;
  if (failed(hlo::verifyReducerShape(
          this->getLoc(), scatterBlock,
          {RankedTensorType::get({}, sourceType.getElementType())},
          {initValueType},
          /*numInputs=*/1, /*allowedDimensions=*/{},
          /*allInputsUnranked=*/false, accumulatorSubshapes)))
    return failure();

  // P3.
  // TODO: add missing tests of hlo::convert1DAttribute( for SelectAndScatterOp.
  auto windowDimsOrErr = hlo::convert1DAttribute(getWindowDimensions(),
                                                 getLoc(), "window_dimensions");
  if (failed(windowDimsOrErr)) return failure();
  if (operandType.hasRank()) {
    if (operandType.getRank() !=
        static_cast<int64_t>((*windowDimsOrErr).size()))
      return emitOpError()
             << "expects window-dimensions size == operand rank, but got "
                "window-dimensions size: "
             << (*windowDimsOrErr).size()
             << " and operand-type: " << operandType
             << " with rank = " << operandType.getRank() << ".";
  }

  // P4.
  auto paddingOrErr = hlo::convertPaddingAttribute(getPadding(), getLoc());
  if (failed(paddingOrErr)) return failure();

  // TODO: add missing tests of hlo::convert1DAttribute( for SelectAndScatterOp.
  auto windowStridesOrErr =
      hlo::convert1DAttribute(getWindowStrides(), getLoc(), "window_strides");
  if (failed(windowStridesOrErr)) return failure();
  auto windowOrErr = hlo::verifyWindowAttributesAndInferWindowDimensions(
      *windowDimsOrErr, *windowStridesOrErr, *paddingOrErr,
      /*lhsDilation=*/{}, /*rhsDilation=*/{}, getLoc());
  if (failed(windowOrErr)) return failure();

  // P5.
  if (!hlo::compatibleShapeAndElementType(operandType, resultType))
    return emitOpError()
           << "expects the return-type to match the operand-type, but got "
           << resultType << " and " << operandType << " resp.";

  // P6.
  auto windowResultType =
      inferSelectAndScatterOpReturnType(operandType, *windowOrErr);

  if (!hlo::compatibleShapeAndElementType(windowResultType, sourceType,
                                          /*ignoreFpPrecision=*/true))
    return emitOpError() << "expects source-type to be " << windowResultType
                         << ", but got" << sourceType;

  return success();
}

//===----------------------------------------------------------------------===//
// ScatterOp
//===----------------------------------------------------------------------===//

/*
 * We intend to verify the following properties:
 * P1. The 'update_window_dims' must be valid indices of 'updates' tensor.
 * P2. The 'inserted_window_dims' must be valid indices of 'operand' tensor.
 * P3. Check if the rank-of('operand') == size-of('update_window_dims') +
 *     size-of('inserted_window_dims')
 * P4. size-of('scatter_dims_to_operand_dims') =
 *         'scatter_indices'['index_vector_dim'] &
 *     'scatter_dims_to_operand_dims' must be valid indices of 'operand' tensor.
 */
LogicalResult validateScatterDimensionNumbers(
    ShapedType operandType, ArrayRef<int64_t> scatterIndicesShape,
    ShapedType updateType, bool operandTypeRanked,
    bool scatterIndicesTypeRanked, bool updatesTypeRanked,
    ScatterDimensionNumbersAttr dimNumbers, Location loc) {
  // P1.
  auto updateWindowDims = to_vector(dimNumbers.getUpdateWindowDims());
  if (!llvm::is_sorted(updateWindowDims))
    return mlir::emitError(loc)
           << "Expects update_window_dims to be sorted; got: ["
           << updateWindowDims << "].";

  if (hasDuplicates(updateWindowDims))
    return mlir::emitError(loc)
           << "Expects update_window_dims to not repeat; got: ["
           << updateWindowDims << "].";

  if (updatesTypeRanked) {
    for (int64_t windowDim : updateWindowDims) {
      if (windowDim < 0 || windowDim >= updateType.getRank()) {
        return mlir::emitError(loc)
               << "Expects each element of update_window_dims to be in range "
                  "[0, "
                  "rank-of('updates') i.e. [0, "
               << updateType.getRank() << "). got: " << windowDim << ".";
      }
    }
  }

  // P2.
  auto insertedWindowDims = to_vector(dimNumbers.getInsertedWindowDims());
  if (!llvm::is_sorted(insertedWindowDims))
    return mlir::emitError(loc)
           << "Expects inserted_window_dims to be sorted; got: ["
           << insertedWindowDims << "].";

  if (hasDuplicates(insertedWindowDims))
    return mlir::emitError(loc)
           << "Expects inserted_window_dims to not repeat; got: ["
           << insertedWindowDims << "].";

  if (operandTypeRanked) {
    for (int64_t insertedDim : insertedWindowDims) {
      if (insertedDim < 0 || insertedDim >= operandType.getRank()) {
        return mlir::emitError(loc)
               << "Expects each element of inserted_window_dims to be in range "
                  "[0, rank-of('operand') i.e. [0, "
               << operandType.getRank() << "). got: " << insertedDim << ".";
      }
    }
  }

  // P3.
  if (operandTypeRanked) {
    auto windowSize = updateWindowDims.size() + insertedWindowDims.size();
    if (operandType.getRank() != static_cast<int64_t>(windowSize))
      return mlir::emitError(loc)
             << "Expects rank-of operand to match "
                "size-of('update_window_dims')  + "
                "size-of('inserted_window_dims') i.e. "
             << windowSize << " but got " << operandType.getRank() << ".";
  }

  // P4.
  auto scatterDimsToOperandDims =
      to_vector(dimNumbers.getScatterDimsToOperandDims());
  auto indexVectorDim = dimNumbers.getIndexVectorDim();
  if (scatterIndicesTypeRanked) {
    if (!hlo::isDynamicDimSize(scatterIndicesShape[indexVectorDim]) &&
        static_cast<int64_t>(scatterDimsToOperandDims.size()) !=
            scatterIndicesShape[dimNumbers.getIndexVectorDim()])
      return mlir::emitError(loc)
             << "Scatter op has " << scatterDimsToOperandDims.size()
             << " elements in scatter_dims_to_operand_dims and the bound of "
                "dimension index_vector_dim="
             << dimNumbers.getIndexVectorDim() << " of scatter_indices is "
             << scatterIndicesShape[dimNumbers.getIndexVectorDim()]
             << ". These two numbers must be equal.";
  }

  if (operandTypeRanked) {
    for (int64_t i = 0;
         i < static_cast<int64_t>(scatterDimsToOperandDims.size()); ++i) {
      int64_t scatterDimToOperandDim = scatterDimsToOperandDims[i];
      if (scatterDimToOperandDim < 0 ||
          scatterDimToOperandDim >= operandType.getRank())
        return mlir::emitError(loc)
               << "Invalid scatter_dims_to_operand_dims mapping; domain is [0, "
               << operandType.getRank() << "), got: " << i << "->"
               << scatterDimToOperandDim << ".";
    }
  }

  if (hasDuplicates(scatterDimsToOperandDims))
    return mlir::emitError(loc)
           << "Expects scatter_dims_to_operand_dims to not repeat; got: ["
           << scatterDimsToOperandDims << "].";

  return success();
}
/*
 * We intend to verify the following properties:
 *  P0. scatter_indices argument must be an integral tensor. Enforced by ODS.
 *  P1. Scatter index leaf dimension must be within [0, rank(scatter_indices)"
 *      " + 1).
 *  P2. Verify reducer shape.
 *  P3. rank-of('updates[i]') == size-of('update_window_dims') +
 *      rank-of('scatter_indices') - 1, where 'scatter_indices' is expanded by a
 *      trailing 1 dimension if 'index_vector_dim' == rank-of('scatter_indices')
 *      for all values of `i`.
 *  P4. Validate the scatter-dimensions-numbers.
 *  P5. Valide the bounds of each of the 'updates' w.r.t the operands.
 *  P6. Validate the bounds of each of the 'updates' w.r.t the
 * 'scatter_indices'.
 *  P7. Check return types.
 */
LogicalResult ScatterOp::verify() {
  // Get the first operand and update, since variadic Scatter is not yet
  // implemented
  auto numOperands = getInputs().size();
  auto scatterIndicesType =
      getScatterIndices().getType().dyn_cast<TensorType>();

  SmallVector<TensorType, 1> operandTypes =
      llvm::to_vector(llvm::map_range(getInputs().getTypes(), [](Type type) {
        return type.cast<TensorType>();
      }));
  SmallVector<TensorType, 1> updatesTypes =
      llvm::to_vector(llvm::map_range(getUpdates().getTypes(), [](Type type) {
        return type.cast<TensorType>();
      }));
  bool allOperandTypesRanked =
      llvm::all_of(getInputs().getTypes(),
                   [](Type type) { return type.isa<RankedTensorType>(); });
  bool scatterIndicesTypeRanked = scatterIndicesType.isa<RankedTensorType>();

  // P1.
  int64_t indexVectorDim = getScatterDimensionNumbers().getIndexVectorDim();
  if (scatterIndicesTypeRanked) {
    if (indexVectorDim > scatterIndicesType.getRank() || indexVectorDim < 0)
      return emitOpError()
             << "expects scatter index leaf dimension to be within [0, "
                "rank(scatter_indices) + 1."
                " rank(scatter_indices) is "
             << scatterIndicesType.getRank()
             << " and scatter index leaf dimension is " << indexVectorDim
             << ".";
  }

  // P2.
  Block& block = getUpdateComputation().front();
  SmallVector<TensorType> accumulatorSubshapes;
  SmallVector<TensorType> inputTypes, initValueTypes;
  for (int64_t i = 0; i < static_cast<int64_t>(numOperands); i++) {
    inputTypes.push_back(operandTypes[i]);
    initValueTypes.push_back(
        RankedTensorType::get({}, updatesTypes[i].getElementType()));
  }
  if (failed(hlo::verifyReducerShape(
          this->getLoc(), block, inputTypes, initValueTypes, numOperands,
          /*allowedDimensions=*/{},
          /*allInputsUnranked=*/!allOperandTypesRanked, accumulatorSubshapes)))
    return failure();

  // P3.
  auto updateWindowDims = getScatterDimensionNumbers().getUpdateWindowDims();
  SmallVector<int64_t> expandedScatterIndicesShape;
  if (scatterIndicesTypeRanked) {
    expandedScatterIndicesShape =
        llvm::to_vector(scatterIndicesType.getShape());
    if (static_cast<int64_t>(expandedScatterIndicesShape.size()) ==
        indexVectorDim)
      expandedScatterIndicesShape.push_back(1);
  }

  for (int64_t i = 0; i < static_cast<int64_t>(numOperands); i++) {
    if (scatterIndicesTypeRanked && updatesTypes[i].isa<RankedTensorType>()) {
      int64_t expectedUpdatesRank =
          expandedScatterIndicesShape.size() - 1 + updateWindowDims.size();
      if (updatesTypes[i].getRank() != expectedUpdatesRank)
        return emitOpError()
               << "expects updates tensor must be of rank "
               << expectedUpdatesRank
               << " ( == rank-of('scatter_indices') - 1 + "
                  "size-of('update_window_dims'), where 'scatter_indices' is "
                  "expanded by a trailing 1 dimension if 'index_vector_dim' == "
                  "rank-of('scatter_indices')), but got "
               << updatesTypes[i].getRank() << ".";
    }
  }

  // P4.
  for (int64_t i = 0; i < static_cast<int64_t>(numOperands); i++) {
    if (failed(validateScatterDimensionNumbers(
            operandTypes[i], expandedScatterIndicesShape, updatesTypes[i],
            operandTypes[i].isa<RankedTensorType>(), scatterIndicesTypeRanked,
            updatesTypes[i].isa<RankedTensorType>(),
            getScatterDimensionNumbers(), getLoc())))
      return failure();
  }

  // P5.
  for (int64_t i = 0; i < static_cast<int64_t>(numOperands); i++) {
    if (updatesTypes[i].isa<RankedTensorType>()) {
      auto updatesShape = updatesTypes[i].getShape();
      if (operandTypes[i].isa<RankedTensorType>()) {
        auto operandShape = operandTypes[i].getShape();
        auto insertedWindowDims =
            getScatterDimensionNumbers().getInsertedWindowDims();

        int64_t insertedDimsSeen = 0;
        SmallVector<int64_t> maxUpdateSliceSizes;
        const auto dimensionsSize = operandTypes[i].getRank();
        maxUpdateSliceSizes.reserve(dimensionsSize);
        for (int i = 0; i < dimensionsSize; ++i) {
          if (insertedDimsSeen <
                  static_cast<int64_t>(insertedWindowDims.size()) &&
              insertedWindowDims[insertedDimsSeen] == i) {
            ++insertedDimsSeen;
          } else {
            maxUpdateSliceSizes.push_back(operandShape[i]);
          }
        }

        for (int64_t i = 0; i < static_cast<int64_t>(updateWindowDims.size());
             ++i) {
          auto updateWindowDim = updateWindowDims[i];

          if (hlo::isDynamicDimSize(updatesShape[updateWindowDim]) ||
              hlo::isDynamicDimSize(maxUpdateSliceSizes[i]))
            continue;

          if (updatesShape[updateWindowDim] > maxUpdateSliceSizes[i]) {
            return emitOpError()
                   << "expects bounds of the window dimensions of "
                      "updates to not exceed the "
                      "bounds of the corresponding dimensions of "
                      "operand. For dimension "
                   << updateWindowDim << ", updates bound is "
                   << updatesShape[updateWindowDim] << ", operand bound is "
                   << maxUpdateSliceSizes[i] << ".";
          }
        }
      }

      // P6.
      if (scatterIndicesTypeRanked) {
        int64_t scatterDimsSeen = 0;
        for (int64_t i = 0; i < static_cast<int64_t>(updatesShape.size());
             ++i) {
          bool isUpdateWindowDim = std::binary_search(
              updateWindowDims.begin(), updateWindowDims.end(), i);

          if (isUpdateWindowDim) continue;
          if (scatterDimsSeen == indexVectorDim) ++scatterDimsSeen;

          if (!hlo::isDynamicDimSize(updatesShape[i]) &&
              !hlo::isDynamicDimSize(
                  expandedScatterIndicesShape[scatterDimsSeen]) &&
              (updatesShape[i] !=
               expandedScatterIndicesShape[scatterDimsSeen])) {
            return emitOpError()
                   << "expects bounds of the scatter dimensions of "
                      "updates to be same as the "
                      "bounds of the corresponding dimensions of "
                      "scatter indices. For "
                      "scatter dimension "
                   << i << ", updates bound is " << updatesShape[i]
                   << " , scatter_indices "
                      "bound is "
                   << expandedScatterIndicesShape[scatterDimsSeen] << ".";
          }
          ++scatterDimsSeen;
        }
      }
    }
  }

  // P7.
  for (int64_t i = 0; i < static_cast<int64_t>(numOperands); i++) {
    if (!hlo::compatibleShapeAndElementType(operandTypes[i],
                                            getResult(i).getType()))
      return emitOpError()
             << "expects the return type to be same as the operand type: "
             << operandTypes[i] << ", but got " << getResult(i).getType()
             << ".";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// WhileOp
//===----------------------------------------------------------------------===//

LogicalResult WhileOp::inferReturnTypes(
    MLIRContext*, Optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  WhileOp::Adaptor adaptor(operands, attributes, regions);
  return hlo::inferWhileOp(location, adaptor.getOperand(), adaptor.getCond(),
                           adaptor.getBody(), inferredReturnTypes);
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
    MLIRContext*, Optional<Location> /*location*/, ValueShapeRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  UniformDequantizeOp::Adaptor adaptor(operands, attributes, regions);
  auto operandType = (*operands.begin()).getType().cast<ShapedType>();
  // Trait HLO_QuantizedIntTensor in ODS guarantees QuantizedType;
  auto quantType = operandType.getElementType().cast<quant::QuantizedType>();
  auto shape = operandType.dyn_cast<ShapedType>().getShape();
  inferredReturnShapes.emplace_back(shape, quantType.getExpressedType());
  return success();
}

}  // namespace stablehlo
}  // namespace mlir

// clang-format off
using mlir::hlo::printSameOperandsAndResultType;
using mlir::hlo::parseSameOperandsAndResultType;
using mlir::hlo::printVariadicSameOperandsAndResultType;
using mlir::hlo::parseVariadicSameOperandsAndResultType;
using mlir::hlo::printVariadicOperandWithAttribute;
using mlir::hlo::parseVariadicOperandWithAttribute;
using mlir::hlo::printComplexOpType;
using mlir::hlo::parseComplexOpType;
using mlir::hlo::printPairwiseOpType;
using mlir::hlo::parsePairwiseOpType;
using mlir::hlo::printSelectOpType;
using mlir::hlo::parseSelectOpType;
using mlir::hlo::printTupleOpType;
using mlir::hlo::parseTupleOpType;
using mlir::hlo::printCustomCallTarget;
using mlir::hlo::parseCustomCallTarget;
using mlir::hlo::printDenseI64Array;
using mlir::hlo::parseDenseI64Array;
using mlir::hlo::printExponentMantissa;
using mlir::hlo::parseExponentMantissa;
// clang-format on

#define GET_OP_CLASSES
#include "stablehlo/dialect/StablehloOps.cpp.inc"

namespace mlir {
namespace stablehlo {

//===----------------------------------------------------------------------===//
// StableHLO Dialect Interfaces
//===----------------------------------------------------------------------===//

namespace {
struct HLOInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  // Allow all call operations to be inlined.
  bool isLegalToInline(Operation* call, Operation* callable,
                       bool wouldBeCloned) const final {
    return true;
  }
  // We don't have any special restrictions on what can be inlined into
  // destination regions (e.g. while/conditional bodies). Always allow it.
  bool isLegalToInline(Region* dest, Region* src, bool wouldBeCloned,
                       BlockAndValueMapping& valueMapping) const final {
    return true;
  }
  // Operations in StableHLO dialect are always legal to inline since they are
  // pure.
  bool isLegalToInline(Operation*, Region*, bool,
                       BlockAndValueMapping&) const final {
    return true;
  }
};

struct HLOBoundedDialectInterface : public hlo::BoundedDialectInterface {
  using BoundedDialectInterface::BoundedDialectInterface;

  Attribute createBoundedAttr(ArrayRef<int64_t> bounds) const override {
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
  addInterfaces<HLOInlinerInterface>();
  addInterfaces<HLOBoundedDialectInterface>();
  addBytecodeInterface(this);
  addTypes<TokenType>();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "stablehlo/dialect/StablehloAttrs.cpp.inc"
      >();
  context->loadDialect<tensor::TensorDialect>();
}

Type StablehloDialect::parseType(DialectAsmParser& parser) const {
  StringRef dataType;
  if (parser.parseKeyword(&dataType)) return Type();

  if (dataType == "token") return TokenType::get(getContext());
  parser.emitError(parser.getNameLoc())
      << "unknown stablehlo type: " << dataType;
  return nullptr;
}

void StablehloDialect::printType(Type type, DialectAsmPrinter& os) const {
  if (type.isa<TokenType>()) {
    os << "token";
    return;
  }
  os << "<unknown stablehlo type>";
}

// Entry point for Attribute parsing, TableGen generated code will handle the
// dispatch to the individual classes.
Attribute StablehloDialect::parseAttribute(DialectAsmParser& parser,
                                           Type type) const {
  StringRef attrTag;
  Attribute attr;
  auto parseResult = generatedAttributeParser(parser, &attrTag, type, attr);
  if (parseResult.has_value()) return attr;
  parser.emitError(parser.getNameLoc(), "unknown stablehlo attribute");
  return Attribute();
}

// Entry point for Attribute printing, TableGen generated code will handle the
// dispatch to the individual classes.
void StablehloDialect::printAttribute(Attribute attr,
                                      DialectAsmPrinter& os) const {
  LogicalResult result = generatedAttributePrinter(attr, os);
  (void)result;
  assert(succeeded(result));
}

/// Helpers for attributes parsing.

static ParseResult parseDims(AsmParser& parser, SmallVector<int64_t>& dims) {
  dims.clear();
  return parser.parseCommaSeparatedList(AsmParser::Delimiter::Square, [&]() {
    return parser.parseInteger(dims.emplace_back());
  });
}

static ParseResult parseDimsWithMinimumElements(AsmParser& parser,
                                                SmallVector<int64_t>& dims,
                                                int minElements) {
  if (failed(parseDims(parser, dims))) return failure();
  if (static_cast<int64_t>(dims.size()) < minElements)
    return parser.emitError(parser.getCurrentLocation())
           << "expected at least " << minElements << " element(s), found "
           << dims.size();
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
      if (succeeded(parser.parseOptionalKeyword(keyword))) {
        if (seen[index]) {
          return parser.emitError(parser.getCurrentLocation())
                 << "duplicated `" << keyword << "` entry";
        }
        if (parseEqual.empty() || parseEqual[index]) {
          if (failed(parser.parseEqual())) return failure();
        }
        if (failed(parseFuncs[index]())) return failure();
        if (failed(parser.parseOptionalComma())) return parser.parseGreater();
        seen[index] = true;
        foundOne = true;
      }
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
  llvm_unreachable("Unknown NonSpatialDim");
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
             non_spatialDims) {
          dims[nonSpatialDim.first] = nonSpatialDim.second;
        }
        for (auto spatial_dim : llvm::enumerate(spatialDims)) {
          dims[spatial_dim.value()] = static_cast<int64_t>(spatial_dim.index());
        }

        // Each dimension numbers will be printed as a comma separated list
        // surrounded by square brackets, e.g., [b, 0, 1, 2, f]
        p << '[';
        llvm::interleaveComma(dims, p, [&](int64_t dim) {
          if (dim >= 0) {
            p << dim;
          } else {
            p << nonSpatialDimToString(static_cast<NonSpatialDim>(dim));
          }
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
    if (parser.parseLSquare()) {
      return failure();
    }

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
        if (parseResult.value().failed()) {
          return failure();
        }
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
        if (parser.parseKeyword(&keyword)) {
          return failure();
        }
        if (keyword.size() != 1 || allowedNonSpatialDims.empty()) {
          return parser.emitError(dimLocation, "Unexpected keyword ")
                 << keyword;
        }
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
    if (parser.parseRSquare()) {
      return failure();
    }

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
  if (parseDims({IOBatch, IOFeature}, parsedDims)) {
    return failure();
  }
  llvm::SmallVector<int64_t> inputSpatialDimensions = parsedDims.first;
  int64_t inputBatchDimension = parsedDims.second[IOBatch];
  int64_t inputFeatureDimension = parsedDims.second[IOFeature];
  if (parser.parseKeyword("x")) return failure();
  if (parseDims({KIFeature, KOFeature}, parsedDims)) {
    return failure();
  }
  llvm::SmallVector<int64_t> kernelSpatialDimensions = parsedDims.first;
  int64_t kernelInputFeatureDimension = parsedDims.second[KIFeature];
  int64_t kernelOutputFeatureDimension = parsedDims.second[KOFeature];
  if (parser.parseArrow()) {
    return failure();
  }
  if (parseDims({IOBatch, IOFeature}, parsedDims)) {
    return failure();
  }
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
  if (!getResultTupleIndices().empty()) {
    printer << ", " << getResultTupleIndices();
  }
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
                           llvm::Optional<DenseIntElementsAttr> windowStrides,
                           llvm::Optional<DenseIntElementsAttr> padding,
                           llvm::Optional<DenseIntElementsAttr> lhsDilation,
                           llvm::Optional<DenseIntElementsAttr> rhsDilation,
                           llvm::Optional<DenseElementsAttr> windowReversal) {
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
    if (!allowedAttributeNames.erase(attributeName)) {
      return parser.emitError(parser.getCurrentLocation(),
                              "Unexpected keyword ")
             << attributeName;
    }

    if (parser.parseEqual()) {
      return failure();
    }

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
                                         innerParser)) {
        return failure();
      }
      const int64_t size = static_cast<int64_t>(values.size());
      // values should be filled with the Nx2 padding values.
      assert(size % 2 == 0);
      auto ty = RankedTensorType::get({size / 2, 2},
                                      parser.getBuilder().getIntegerType(64));
      padding = DenseIntElementsAttr::get(ty, values);
    } else {
      // Parse 1D array of integers.
      if (parser.parseCommaSeparatedList(AsmParser::Delimiter::Square,
                                         int64Parser)) {
        return failure();
      }
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
          llvm_unreachable("Unexpected attribute name");
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
                                    llvm::Optional<StringRef> compareType,
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
  llvm::Optional<StringRef> compareType = llvm::None;
  for (auto const& elementType : elementTypes)
    if (elementType.isa<FloatType>()) {
      compareType.emplace("TOTALORDER");
      break;
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
  if (auto aliasAttr = attr.getValue().dyn_cast<ArgResultAliasAttr>()) {
    if (failed(
            verifyArgResultAliasAttr(attr.getName(), aliasAttr, argIndex, op)))
      return failure();
  }
  return success();
}

LogicalResult StablehloDialect::verifyOperationAttribute(Operation* op,
                                                         NamedAttribute attr) {
  if (auto aliasAttr = attr.getValue().dyn_cast<ArgResultAliasAttr>()) {
    if (!isa<mlir::FunctionOpInterface>(op))
      return op->emitOpError()
             << "attribute " << attr.getName()
             << " can only be used on function-like operations";
  }
  return success();
}

}  // namespace stablehlo
}  // namespace mlir
