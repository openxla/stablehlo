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

#include "stablehlo/dialect/Base.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <tuple>
#include <utility>

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/TypeID.h"

// Include order matters
#include "stablehlo/dialect/BaseAttrInterfaces.cpp.inc"

#define DEBUG_TYPE "stablehlo-base"

namespace mlir {
namespace hlo {

LogicalResult verifyCompatibleShapeWithBounds(Type type1, Type type2) {
  if (failed(verifyCompatibleShape(type1, type2))) return failure();

  // Verify shapes against bounds
  auto isCompatible = [](ArrayRef<int64_t> shape,
                         BoundedAttrInterface boundedAttr) {
    if (shape.empty() || !boundedAttr) return true;
    auto bounds = boundedAttr.getBounds();
    for (auto [dim_size, bound] : llvm::zip(shape, bounds))  // NOLINT
      if (!isDynamicDimSize(bound) && bound < dim_size) return false;
    return true;
  };

  RankedTensorType rankedType1 = dyn_cast<RankedTensorType>(type1);
  RankedTensorType rankedType2 = dyn_cast<RankedTensorType>(type2);
  if (rankedType1 && rankedType2) {
    auto boundedAttr1 =
        dyn_cast_or_null<BoundedAttrInterface>(rankedType1.getEncoding());
    auto boundedAttr2 =
        dyn_cast_or_null<BoundedAttrInterface>(rankedType2.getEncoding());
    return LogicalResult::success(
        isCompatible(rankedType1.getShape(), boundedAttr2) &&
        isCompatible(rankedType2.getShape(), boundedAttr1));
  }
  return success();
}

bool isCompatibleElementTypeForHloTypeInference(Type tp1, Type tp2) {
  // Get element type if shaped
  tp1 = getElementTypeOrSelf(tp1);
  tp2 = getElementTypeOrSelf(tp2);

  // For quantized types:
  //   a. both `tp1` and `tp2` should be quantized types
  //   b. with similar quantization granularity (i.e. both per-tensor or both
  //   per-axis)
  //   c. with equal storage_type, storage_type_min, storage_type_max, and
  //   expressed_type
  auto qtp1 = dyn_cast<quant::QuantizedType>(tp1);
  auto qtp2 = dyn_cast<quant::QuantizedType>(tp2);
  if (qtp1 && qtp2) {
    if (qtp1.getStorageType() != qtp2.getStorageType() ||
        qtp1.getStorageTypeMin() != qtp2.getStorageTypeMin() ||
        qtp1.getStorageTypeMax() != qtp2.getStorageTypeMax() ||
        qtp1.getExpressedType() != qtp2.getExpressedType()) {
      return false;
    }

    auto qpatp1 = dyn_cast<quant::UniformQuantizedPerAxisType>(qtp1);
    auto qpatp2 = dyn_cast<quant::UniformQuantizedPerAxisType>(qtp2);
    bool quantizationGranularityMatches =
        (qpatp1 && qpatp2) || (!qpatp1 && !qpatp2);

    return quantizationGranularityMatches;
  }

  // return false if only one is of quantized type
  if (qtp1 || qtp2) return false;

  // Sparsity: In the most general case, we allow any combination of
  // sparsity/denseness across any combination of operands/results, as well as
  // differences in sparsity encodings for operands and results.
  // Individual ops may introduce additional constraints.
  // No additional code is needed to check this because of how sparsity is
  // currently implemented.

  // Default case: Unless dynamism, quantization and/or sparsity are involved,
  // the types are required to be exactly equal.
  return tp1 == tp2;
}

bool isCompatibleForHloTypeInference(Type tp1, Type tp2) {
  // Dynamism: We don't require shapes to be the same, we only require them
  // to be compatible, which means that:
  //   1) At least one of the shapes is unranked.
  //   2) Or both shapes have the same rank and their dimensions are compatible,
  //     i.e. for each pair of corresponding dimensions:
  //       2.1) At least one of the dimensions is dynamic,
  //       2.2) Or both dimensions are equal.
  // These relaxed rules simplify the implementation of type inference, allowing
  // ops with partially inferred types to pass verification.
  auto stp1 = dyn_cast<ShapedType>(tp1);
  auto stp2 = dyn_cast<ShapedType>(tp2);
  if (stp1 && stp2)
    return succeeded(verifyCompatibleShapeWithBounds(stp1, stp2)) &&
           isCompatibleElementTypeForHloTypeInference(stp1.getElementType(),
                                                      stp2.getElementType());

  return isCompatibleElementTypeForHloTypeInference(tp1, tp2);
}

bool isCompatibleForHloTypeInference(TypeRange tp1, TypeRange tp2) {
  if (tp1.size() != tp2.size()) return false;
  for (auto [lt, rt] : llvm::zip(tp1, tp2))
    if (!isCompatibleForHloTypeInference(lt, rt)) return false;
  return true;
}

bool isCompatibleForHloTypeInference(ArrayRef<int64_t> shape1, Type tp2) {
  if (llvm::any_of(shape1, [&](int64_t x) { return x < 0; })) return false;
  auto stp2 = dyn_cast<ShapedType>(tp2);
  if (!stp2) return false;
  return isCompatibleForHloTypeInference(
      RankedTensorType::get(shape1, stp2.getElementType()), tp2);
}

bool isCompatibleForHloTypeInference(Value shape1, Type tp2) {
  SmallVector<int64_t> shapeVec1;
  if (!succeeded(matchInts(shape1, shapeVec1))) return true;
  return isCompatibleForHloTypeInference(shapeVec1, tp2);
}

LogicalResult matchInt(Value value, int64_t& result) {
  APInt constValue;
  if (!matchPattern(value, m_ConstantInt(&constValue))) return failure();
  result = constValue.getSExtValue();
  return success();
}

LogicalResult matchInts(Value value, SmallVector<int64_t>& result) {
  DenseIntElementsAttr attr;
  if (!matchPattern(value, m_Constant(&attr))) return failure();
  for (auto element : attr.getValues<APInt>()) {
    result.push_back(element.getSExtValue());
  }
  return success();
}

LogicalResult matchInts(Value value, SmallVector<APSInt>& result) {
  DenseIntElementsAttr attr;
  if (!matchPattern(value, m_Constant(&attr))) return failure();

  // Signless types are treated as signed, per StableHLO convention.
  // Unless the type is i1 (which models boolean type from the StableHLO spec),
  // in which case it's considered to be unsigned.
  auto elementType = attr.getType().getElementType();
  auto isUnsigned = elementType.isUnsignedInteger() ||
                    elementType.getIntOrFloatBitWidth() == 1;

  for (auto element : attr.getValues<APInt>()) {
    result.push_back(APSInt(element, /*isUnsigned=*/isUnsigned));
  }
  return success();
}

LogicalResult matchInts(Value value) {
  DenseIntElementsAttr attr;
  return success(/*isSuccess=*/matchPattern(value, m_Constant(&attr)));
}

LogicalResult deriveShapeFromOperand(
    OpBuilder* builder, Operation* op, Value operand,
    SmallVectorImpl<Value>* reifiedReturnShapes) {
  auto shapedTy = dyn_cast<ShapedType>(operand.getType());
  if (!shapedTy) {
    op->emitOpError() << "operand is not a shaped type";
    return failure();
  }
  reifiedReturnShapes->assign(
      {builder->create<shape::ShapeOfOp>(op->getLoc(), operand)});
  return success();
}

ShapedType getSameShapeTensorType(ShapedType shapedType, Type elementType) {
  if (isa<TensorType>(shapedType)) return shapedType.clone(elementType);
  llvm::report_fatal_error("unsupported type");
}

// createRealType takes a tensor type that may have complex elements and
// returns a type that maintains the shape, but with real numeric data types.
//   Ex: tensor<4xcomplex<f32>>  -->  tensor<4xf32>
ShapedType createRealType(ShapedType type) {
  auto elementTy = type.getElementType();
  if (auto complexTy = dyn_cast<ComplexType>(elementTy))
    elementTy = complexTy.getElementType();
  return hlo::getSameShapeTensorType(type, elementTy);
}

//===----------------------------------------------------------------------===//
// Utils for shape functions with bounded dynamism.
//===----------------------------------------------------------------------===//

LogicalResult verifyBounds(ArrayRef<int64_t> bounds, RankedTensorType type,
                           function_ref<InFlightDiagnostic()> emitError) {
  int64_t boundsLen = bounds.size();
  int64_t rank = type.getRank();
  if (boundsLen != rank)
    return emitError() << "Bounds length is " << boundsLen
                       << ", expected to be equal to rank(" << rank
                       << ") of the tensor";

  for (int64_t dim = 0; dim < rank; ++dim) {
    int64_t bound = bounds[dim];
    int64_t dimSize = type.getDimSize(dim);
    if (bound != ShapedType::kDynamic && dimSize != ShapedType::kDynamic)
      return emitError() << "Static dimension " << dim
                         << " cannot have a bound, use ShapedType::kDynamic to "
                            "indicate a missing bound";
  }

  return success();
}

ArrayRef<int64_t> encodingToBounds(Attribute encoding) {
  if (auto boundedAttr = dyn_cast_or_null<BoundedAttrInterface>(encoding))
    return boundedAttr.getBounds();
  return {};
}

Attribute boundsToEncoding(Attribute prototype, ArrayRef<int64_t> bounds) {
  if (bounds.empty()) return prototype;
  if (llvm::all_of(bounds, [&](auto b) { return isDynamicDimSize(b); }))
    return {};
  if (!prototype)
    llvm::report_fatal_error(
        "Expect an prototype attribute to obtain the underlying dialect but "
        "got none");
  auto dialect = cast<HloDialectInterface>(&prototype.getDialect());
  return dialect->createTypeExtensions(bounds);
}

// Inference rules to concat dimensions with bounds (lhs/rhs are commutative):
//       Dim of lhs     Dim of rhs      Infer
//  c0:  X              Y               X+Y
//  c1:  X              ?               ?
//  c2:  X              ?, B            ?, X+B
//  c3:  ?              ?               ?
//  c4:  ?              ?, B            ?
//  c5:  ?, B           ?, C            ?, B+C
std::pair<int64_t, int64_t> inferConcatenatedDimAndBound(int64_t leftSize,
                                                         int64_t rightSize,
                                                         int64_t leftBound,
                                                         int64_t rightBound) {
  bool isLeftStaticDim = !isDynamicDimSize(leftSize);
  bool isRightStaticDim = !isDynamicDimSize(rightSize);
  int64_t inferredSize = ShapedType::kDynamic;
  int64_t inferredBound = ShapedType::kDynamic;

  if (isLeftStaticDim && isRightStaticDim) {
    inferredSize = leftSize + rightSize;
  } else {
    int64_t leftSizeOrBound = isLeftStaticDim ? leftSize : leftBound;
    int64_t rightSizeOrBound = isRightStaticDim ? rightSize : rightBound;
    if (!isDynamicDimSize(leftSizeOrBound) &&
        !isDynamicDimSize(rightSizeOrBound))
      inferredBound = leftSizeOrBound + rightSizeOrBound;
  }
  return {inferredSize, inferredBound};
}

// Inference rules to merge dimensions with bounds (lhs/rhs are commutative):
//       Dim of lhs     Dim of rhs      Infer
//  c0:  X              X               X
//  c1:  X              ?               X
//  c2:  X              ?, B(>=X)       X
//  c3:  X              ?, B(<X)        Will error out by compatible checks
//  c4:  ?              ?               ?
//  c5:  ?              ?, B            ?, B
//  c6:  ?, B           ?, C            ?, min(B, C)
FailureOr<std::pair<int64_t, int64_t>> inferMostSpecificDimAndBound(
    std::optional<Location> location, int64_t dim, int64_t leftSize,
    int64_t rightSize, int64_t leftBound, int64_t rightBound) {
  bool isLeftStaticDim = !isDynamicDimSize(leftSize);
  bool isRightStaticDim = !isDynamicDimSize(rightSize);
  bool isLeftStaticBound = !isDynamicDimSize(leftBound);
  bool isRightStaticBound = !isDynamicDimSize(rightBound);
  int64_t inferredSize = ShapedType::kDynamic;
  int64_t inferredBound = ShapedType::kDynamic;

  if (isLeftStaticDim || isRightStaticDim) {
    if (isLeftStaticDim && isRightStaticDim && leftSize != rightSize)
      return emitOptionalError(location, "Mismatched dimension sizes ",
                               leftSize, " and ", rightSize, " in dimension ",
                               dim);
    inferredSize = isLeftStaticDim ? leftSize : rightSize;
    if (isLeftStaticBound || isRightStaticBound) {
      int64_t check_bound = isLeftStaticBound ? leftBound : rightBound;
      if (inferredSize > check_bound)
        return emitOptionalError(location, "Mismatched dimension size ",
                                 inferredSize, " and bound ", check_bound,
                                 " in dimension ", dim);
    }
  } else {
    if (isLeftStaticBound && isRightStaticBound)
      inferredBound = std::min(leftBound, rightBound);
    else
      inferredBound = isLeftStaticBound ? leftBound : rightBound;
  }
  return std::make_pair(inferredSize, inferredBound);
}

// Inference rules for conditional branches (lhs/rhs are commutative):
//       Dim of lhs     Dim of rhs      Infer
//  c0:  X              X               X
//  c1:  X              ?               ?
//  c2:  X              ?, B            ?, max(X, B)
//  c3:  ?              ?               ?
//  c4:  ?              ?, B            ?
//  c5:  ?, B           ?, C            ?, max(B, C)
FailureOr<std::pair<int64_t, int64_t>> inferLeastSpecificDimAndBound(
    std::optional<Location> location, int64_t dim, int64_t leftSize,
    int64_t rightSize, int64_t leftBound, int64_t rightBound) {
  bool isLeftStaticDim = !isDynamicDimSize(leftSize);
  bool isRightStaticDim = !isDynamicDimSize(rightSize);
  bool isLeftStaticBound = !isDynamicDimSize(leftBound);
  bool isRightStaticBound = !isDynamicDimSize(rightBound);
  int64_t inferredSize = ShapedType::kDynamic;
  int64_t inferredBound = ShapedType::kDynamic;

  if (isLeftStaticDim || isRightStaticDim) {
    if (isLeftStaticDim && isRightStaticDim) {
      if (leftSize != rightSize)
        return emitOptionalError(location, "Mismatched dimension sizes ",
                                 leftSize, " and ", rightSize, " in dimension ",
                                 dim);
      inferredSize = leftSize;
    } else if (isLeftStaticBound || isRightStaticBound) {
      inferredBound = isLeftStaticDim ? std::max(leftSize, rightBound)
                                      : std::max(rightSize, leftBound);
    }
  } else if (isLeftStaticBound && isRightStaticBound) {
    inferredBound = std::max(leftBound, rightBound);
  }
  return std::make_pair(inferredSize, inferredBound);
}

FailureOr<ShapedType> inferTypeWithCustomFn(
    std::optional<Location> location, SmallVector<RankedTensorType> rankedTypes,
    std::function<FailureOr<std::pair<int64_t, int64_t>>(
        std::optional<Location>, int64_t, int64_t, int64_t, int64_t, int64_t)>
        inferDimAndBoundFn) {
  auto rank = rankedTypes[0].getRank();
  for (auto& type : rankedTypes) {
    if (type.getRank() != rank) {
      return emitOptionalError(location, "Mismatched ranks of types",
                               rankedTypes[0].getRank(), " vs ",
                               type.getRank());
    }
  }
  SmallVector<int64_t> inferredSizes = to_vector(rankedTypes[0].getShape());
  SmallVector<int64_t> inferredBounds(rank, ShapedType::kDynamic);
  ArrayRef<int64_t> bounds = encodingToBounds(rankedTypes[0].getEncoding());
  if (!bounds.empty()) inferredBounds = to_vector(bounds);
  bool anyInputHaveBounds = !bounds.empty();

  for (unsigned i = 1; i < rankedTypes.size(); ++i) {
    bounds = encodingToBounds(rankedTypes[i].getEncoding());
    for (int dim = 0; dim < rank; ++dim) {
      auto inferredDimAndBoundOrErr = inferDimAndBoundFn(
          location, dim,
          /*leftSize=*/inferredSizes[dim],
          /*rightSize=*/rankedTypes[i].getShape()[dim],
          /*leftBound=*/inferredBounds[dim],
          /*rightBound=*/bounds.empty() ? ShapedType::kDynamic : bounds[dim]);
      if (failed(inferredDimAndBoundOrErr)) return failure();
      inferredSizes[dim] = (*inferredDimAndBoundOrErr).first;
      inferredBounds[dim] = (*inferredDimAndBoundOrErr).second;
    }
  }

  return {RankedTensorType::get(
      inferredSizes, rankedTypes[0].getElementType(),
      boundsToEncoding(
          rankedTypes[0].getEncoding(),
          // Empty array as argument is an indicator to boundsToEncoding() that
          // there are no bounds at all in inputs, thus sparsity attributes will
          // be included in the return type
          anyInputHaveBounds ? inferredBounds : ArrayRef<int64_t>({})))};
}

FailureOr<Type> inferLeastSpecificShapedType(std::optional<Location> location,
                                             TypeRange inputTypes) {
  SmallVector<RankedTensorType> rankedTypes;
  for (auto inputType : inputTypes)
    if (auto rankedType = dyn_cast<RankedTensorType>(inputType))
      rankedTypes.push_back(rankedType);
    else
      return inputType;
  return inferTypeWithCustomFn(location, rankedTypes,
                               inferLeastSpecificDimAndBound);
}

FailureOr<Type> inferMostSpecificShapedType(std::optional<Location> location,
                                            TypeRange inputTypes) {
  SmallVector<RankedTensorType> rankedTypes;
  for (auto inputType : inputTypes)
    if (auto rankedType = dyn_cast<RankedTensorType>(inputType))
      rankedTypes.push_back(rankedType);
  if (rankedTypes.empty()) return inputTypes[0];
  return inferTypeWithCustomFn(location, rankedTypes,
                               inferMostSpecificDimAndBound);
}

// Applies `fn` to `inputTypes`, using `location` for errors.
// If `inputTypes` are tuples, then applies `fn` to them elementwise and
// wraps the results into a tuple, for example:
//   mapOverTupleElements({tuple<T11, T12>, tuple<T21, T22>}, fn) =
//     tuple<fn(T11, T21), fn(T12, T22)>
// Only supports `inputTypes` where either all types are tuples or no types
// are tuples.
FailureOr<Type> mapOverTupleElements(
    std::optional<Location> location, TypeRange inputTypes,
    function_ref<FailureOr<Type>(std::optional<Location>, TypeRange types)>
        fn) {
  SmallVector<TupleType> tupleTypes;
  for (auto inputType : inputTypes) {
    if (auto tupleType = dyn_cast<TupleType>(inputType))
      tupleTypes.push_back(tupleType);
  }
  if (!tupleTypes.empty()) {
    if (tupleTypes.size() != inputTypes.size())
      return emitOptionalError(location,
                               "Mismatched type kinds: either all types ",
                               "must be tuples, or no types must be tuples");
    SmallVector<Type> results(tupleTypes[0].size());
    for (auto tupleType : tupleTypes) {
      if (tupleType.size() != results.size())
        return emitOptionalError(location,
                                 "Mismatched tuple sizes: all tuple sizes ",
                                 "must be the same");
    }
    for (size_t i = 0; i < results.size(); ++i) {
      SmallVector<Type> ithElements;
      for (auto tupleType : tupleTypes)
        ithElements.push_back(tupleType.getType(i));
      auto result = fn(location, ithElements);
      if (failed(result)) return failure();
      results[i] = *result;
    }
    return TupleType::get(tupleTypes[0].getContext(), results);
  }
  return fn(location, inputTypes);
}

FailureOr<Type> inferLeastSpecificType(std::optional<Location> location,
                                       TypeRange inputTypes) {
  return mapOverTupleElements(location, inputTypes,
                              inferLeastSpecificShapedType);
}

FailureOr<Type> inferMostSpecificType(std::optional<Location> location,
                                      TypeRange inputTypes) {
  return mapOverTupleElements(location, inputTypes,
                              inferMostSpecificShapedType);
}

LogicalResult inferMostSpecificTypeComponents(
    std::optional<Location> location, TypeRange inputTypes,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  auto inferredTypeOrErr = inferMostSpecificType(location, inputTypes);
  if (failed(inferredTypeOrErr)) return failure();

  auto rankedResultType = dyn_cast<RankedTensorType>((*inferredTypeOrErr));
  if (!rankedResultType) {
    auto inferredShapeType = dyn_cast<ShapedType>((*inferredTypeOrErr));
    if (!inferredShapeType) return failure();
    inferredReturnShapes.emplace_back(inferredShapeType);
  } else {
    inferredReturnShapes.emplace_back(rankedResultType.getShape(),
                                      rankedResultType.getElementType(),
                                      rankedResultType.getEncoding());
  }

  return success();
}

LogicalResult getShapeRefinements(
    std::optional<Location> location, Operation* op,
    SmallVector<ShapedTypeComponents>& refinements) {
  auto indicesAttr = dyn_cast_or_null<DenseIntElementsAttr>(
      op->getAttr("indices_of_shape_operands"));
  if (!indicesAttr) return failure();

  SmallVector<Type> flattenedResultTypes;
  flattenTupleTypes(op->getResultTypes(), flattenedResultTypes);
  int64_t flattenedNumResults = flattenedResultTypes.size();
  StringRef flattenedErrorMessageSuffix =
      op->getNumResults() != flattenedNumResults ? ", with tuples flattened"
                                                 : "";

  if (indicesAttr.getNumElements() != flattenedNumResults)
    return emitOptionalError(location, "indices_of_shape_operands: number of ",
                             "elements (", indicesAttr.getNumElements(), ") ",
                             "must be equal to the number of operation results",
                             " (", flattenedNumResults, ")",
                             flattenedErrorMessageSuffix);
  if (indicesAttr.getType().getRank() != 1)
    return emitOptionalError(location, "indices_of_shape_operands: must have ",
                             "rank = 1");
  if (!indicesAttr.getType().getElementType().isInteger(64))
    return emitOptionalError(location, "indices_of_shape_operands: must have ",
                             "i64 element type");

  auto resultIndex = 0;
  for (auto [operandIndex, resultType] :
       llvm::zip(indicesAttr.getValues<int64_t>(), flattenedResultTypes)) {
    if (operandIndex < 0 || operandIndex >= op->getNumOperands())
      return emitOptionalError(location, "indices_of_shape_operands: index #",
                               resultIndex, " (", operandIndex, ") ",
                               "must be within bounds for operation operands ",
                               "(from 0 to ", op->getNumOperands(), ")");

    Value operand = op->getOperand(operandIndex);
    SmallVector<int64_t> refinement;
    if (failed(hlo::matchInts(operand, refinement))) return failure();
    if (!isCompatibleForHloTypeInference(operand, resultType))
      return emitOptionalError(
          location, "indices_of_shape_operands: refinement #", resultIndex,
          " ([", refinement, "]) must be compatible with operation result #",
          resultIndex, " (", resultType, ")", flattenedErrorMessageSuffix);
    refinements.emplace_back(refinement);
    ++resultIndex;
  }
  return success();
}

void flattenTupleTypes(TypeRange types, SmallVector<Type>& result) {
  for (auto type : types) {
    if (auto tupleType = dyn_cast<TupleType>(type)) {
      flattenTupleTypes(tupleType.getTypes(), result);
      continue;
    }
    result.push_back(type);
  }
}

LogicalResult unflattenTupleTypes(TypeRange prototype, TypeRange types,
                                  SmallVector<Type>& result) {
  // Recursively unflattens types into result according to the prototype
  // and returns the number of consumed types or a failure if the prototype
  // and the types are incompatible.
  // This specific kind of return value is what enables a recursive formulation
  // of this algorithm which avoids mutable state except for the result.
  std::function<FailureOr<int64_t>(TypeRange, TypeRange, SmallVector<Type>&)>
      loop;
  loop = [&](TypeRange prototype, TypeRange types,
             SmallVector<Type>& result) -> FailureOr<int64_t> {
    if (prototype.empty() || types.empty()) {
      if (prototype.empty() ^ types.empty()) return {};
      return 0;
    }

    if (auto prototypeFront = dyn_cast<TupleType>(prototype.front())) {
      SmallVector<Type> tupleResult;
      auto consumedFront = loop(prototypeFront.getTypes(), types, tupleResult);
      if (failed(consumedFront)) return {};
      auto consumedRest = loop(prototype.drop_front(),
                               types.drop_front(*consumedFront), result);
      if (failed(consumedRest)) return {};
      result.push_back(
          TupleType::get(prototypeFront.getContext(), tupleResult));
      return *consumedFront + *consumedRest;
    }

    result.push_back(types.front());
    auto consumed = loop(prototype.drop_front(), types.drop_front(), result);
    if (failed(consumed)) return {};
    return *consumed + 1;
  };
  auto consumed = loop(prototype, types, result);
  return success(/*succeeded=*/consumed != -1);
}

ShapedType createShapedType(ShapedTypeComponents components) {
  if (!components.getElementType()) return ShapedType();
  return RankedTensorType::get(components.getDims(),
                               components.getElementType(),
                               components.getAttribute());
}

bool isSplatArray(ArrayRef<int64_t> arr, int64_t val) {
  return std::all_of(arr.begin(), arr.end(),
                     [val](int64_t x) { return x == val; });
}

namespace detail {
template <typename LHS, typename RHS, typename Accum, int64_t N>
bool match(Type lhsPrecisionType, Type rhsPrecisionType, Type accumulationType,
           int64_t numPrimitiveOperations) {
  return isa<LHS>(lhsPrecisionType) && isa<RHS>(rhsPrecisionType) &&
         isa<Accum>(accumulationType) && numPrimitiveOperations == N;
}

FailureOr<KnownDotAlgorithm> getKnownDotAlgorithm(
    Type lhsPrecisionType, Type rhsPrecisionType, Type accumulationType,
    int64_t lhsComponentCount, int64_t rhsComponentCount,
    int64_t numPrimitiveOperations, bool allowImpreciseAccumulation) {
  // Only support single component for now.
  if (lhsComponentCount != 1 || rhsComponentCount != 1) return failure();

  auto isAnyF8 = [](Type t) {
    return llvm::isa<Float8E3M4Type, Float8E4M3Type, Float8E4M3FNType,
                     Float8E5M2Type, Float8E4M3FNUZType, Float8E4M3B11FNUZType,
                     Float8E5M2FNUZType>(t);
  };
  if (isAnyF8(lhsPrecisionType) && isAnyF8(rhsPrecisionType) &&
      accumulationType.isF32() && numPrimitiveOperations == 1) {
    if (allowImpreciseAccumulation)
      return KnownDotAlgorithm::ANY_F8_ANY_F8_F32_FAST_ACCUM;
    return KnownDotAlgorithm::ANY_F8_ANY_F8_F32;
  }
  if (allowImpreciseAccumulation) return failure();

  // TypeID doesn't define a `<` operator so cannot use in map.
  // Use its name instead.
  auto key = std::make_tuple(lhsPrecisionType.getAbstractType().getName(),
                             rhsPrecisionType.getAbstractType().getName(),
                             accumulationType.getAbstractType().getName(),
                             numPrimitiveOperations);

  StringRef bf16 = BFloat16Type::name;
  StringRef f16 = Float16Type::name;
  StringRef f32 = Float32Type::name;
  StringRef f64 = Float64Type::name;
  StringRef tf32 = FloatTF32Type::name;
  std::map<std::tuple<StringRef, StringRef, StringRef, int64_t>,
           KnownDotAlgorithm>
      knownDotAlgorithms{
          {{f16, f16, f16, 1}, KnownDotAlgorithm::F16_F16_F16},
          {{f16, f16, f32, 1}, KnownDotAlgorithm::F16_F16_F32},
          {{bf16, bf16, bf16, 1}, KnownDotAlgorithm::BF16_BF16_BF16},
          {{bf16, bf16, f32, 1}, KnownDotAlgorithm::BF16_BF16_F32},
          {{bf16, bf16, f32, 3}, KnownDotAlgorithm::BF16_BF16_F32_X3},
          {{bf16, bf16, f32, 6}, KnownDotAlgorithm::BF16_BF16_F32_X6},
          {{tf32, tf32, f32, 1}, KnownDotAlgorithm::TF32_TF32_F32},
          {{tf32, tf32, f32, 3}, KnownDotAlgorithm::TF32_TF32_F32_X3},
          {{f32, f32, f32, 1}, KnownDotAlgorithm::F32_F32_F32},
          {{f64, f64, f64, 1}, KnownDotAlgorithm::F64_F64_F64},
      };

  auto algorithm = knownDotAlgorithms.find(key);
  if (algorithm != knownDotAlgorithms.end()) {
    LLVM_DEBUG(llvm::dbgs()
               << "Found known dot algorithm: "
               << static_cast<int64_t>(algorithm->second) << " "
               << std::get<0>(key) << ", " << std::get<1>(key) << ", "
               << std::get<2>(key) << ", " << std::get<3>(key) << "\n");
    return algorithm->second;
  }
  return failure();
}
}  // namespace detail

// Check if the combination of a dot algorithm struct is known.
bool isKnownDotAlgorithm(Type lhsPrecisionType, Type rhsPrecisionType,
                         Type accumulationType, int64_t lhsComponentCount,
                         int64_t rhsComponentCount,
                         int64_t numPrimitiveOperations,
                         bool allowImpreciseAccumulation) {
  return succeeded(detail::getKnownDotAlgorithm(
      lhsPrecisionType, rhsPrecisionType, accumulationType, lhsComponentCount,
      rhsComponentCount, numPrimitiveOperations, allowImpreciseAccumulation));
}

mlir::Speculation::Speculatability getShapedSpeculatability(
    Operation* op, int64_t shapeCount) {
  // If all inputs are static and the shape-related operands are constant
  // then any relationship between the input, the shapes and the output can be
  // verified statically.
  bool allInputsStatic = llvm::all_of(op->getOperandTypes(), [](Type t) {
    return cast<ShapedType>(t).hasStaticShape();
  });
  bool allShapesConstant = llvm::all_of(llvm::seq(shapeCount), [&](int64_t i) {
    return matchPattern(op->getOperand(op->getNumOperands() - 1 - i),
                        m_Constant());
  });
  return allInputsStatic && allShapesConstant
             ? mlir::Speculation::Speculatable
             : mlir::Speculation::NotSpeculatable;
}

bool isValidStablehloQuantizedElementType(Type elementType) {
  auto quantizedElementType = dyn_cast<mlir::quant::QuantizedType>(elementType);
  if (!quantizedElementType) return false;

  int64_t storageTypeMin = quantizedElementType.getStorageTypeMin();
  int64_t storageTypeMax = quantizedElementType.getStorageTypeMax();

  SmallVector<int64_t> zeroPoints;
  SmallVector<double> scales;
  if (auto quantizedPerTensorElementType =
          dyn_cast<mlir::quant::UniformQuantizedType>(elementType)) {
    zeroPoints.push_back(quantizedPerTensorElementType.getZeroPoint());
    scales.push_back(quantizedPerTensorElementType.getScale());
  } else {
    auto quantizedPerAxisElementType =
        cast<mlir::quant::UniformQuantizedPerAxisType>(elementType);
    zeroPoints.insert(zeroPoints.begin(),
                      quantizedPerAxisElementType.getZeroPoints().begin(),
                      quantizedPerAxisElementType.getZeroPoints().end());
    scales.insert(scales.begin(),
                  quantizedPerAxisElementType.getScales().begin(),
                  quantizedPerAxisElementType.getScales().end());
  }

  // quantized_type_c7, quantized_type_c8
  if (llvm::any_of(zeroPoints, [&](int64_t zeroPoint) {
        return storageTypeMin > zeroPoint || zeroPoint > storageTypeMax;
      })) {
    return false;
  }

  return true;
}

bool isValidQuantizedDimension(Type type) {
  auto rankedType = dyn_cast<RankedTensorType>(type);
  if (!rankedType) return true;

  auto quantizedPerAxisElementType =
      dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(
          rankedType.getElementType());

  if (!quantizedPerAxisElementType) return true;

  // quantized_type_c12, quantized_type_c13
  int64_t quantDim = quantizedPerAxisElementType.getQuantizedDimension();
  int64_t numScales =
      static_cast<int64_t>(quantizedPerAxisElementType.getScales().size());
  return quantDim < rankedType.getRank() &&
         (!rankedType.isDynamicDim(quantDim) &&
          numScales == rankedType.getDimSize(quantDim));
}

}  // namespace hlo
}  // namespace mlir
