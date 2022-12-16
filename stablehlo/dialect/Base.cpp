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

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Quant/QuantTypes.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LLVM.h"

// Include order matters
#include "stablehlo/dialect/BaseAttrInterfaces.cpp.inc"

namespace mlir {
namespace hlo {

namespace {
Type getExpressedTypeOrSelf(Type type) {
  auto quantType = type.dyn_cast<quant::QuantizedType>();
  return quantType ? quantType.getExpressedType() : type;
}
}  // namespace

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

  RankedTensorType rankedType1 = type1.dyn_cast<RankedTensorType>();
  RankedTensorType rankedType2 = type2.dyn_cast<RankedTensorType>();
  if (rankedType1 && rankedType2) {
    auto boundedAttr1 =
        rankedType1.getEncoding().dyn_cast_or_null<BoundedAttrInterface>();
    auto boundedAttr2 =
        rankedType2.getEncoding().dyn_cast_or_null<BoundedAttrInterface>();
    return LogicalResult::success(
        isCompatible(rankedType1.getShape(), boundedAttr2) &&
        isCompatible(rankedType2.getShape(), boundedAttr1));
  }
  return success();
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
  auto stp1 = tp1.dyn_cast<ShapedType>();
  auto stp2 = tp2.dyn_cast<ShapedType>();
  if (stp1 && stp2) {
    return succeeded(verifyCompatibleShapeWithBounds(stp1, stp2)) &&
           isCompatibleForHloTypeInference(stp1.getElementType(),
                                           stp2.getElementType());
  }

  // Quantization: In the most general case, we allow any combination of
  // quantized/non-quantized across any combination of operands/results,
  // and some differences in quantization parameters across operands/results.
  // Individual ops may introduce additional constraints.
  auto qtp1 = tp1.dyn_cast<quant::QuantizedType>();
  auto qtp2 = tp2.dyn_cast<quant::QuantizedType>();
  if (qtp1 && qtp2) {
    if (qtp1.getStorageType() != qtp2.getStorageType() ||
        qtp1.getStorageTypeMin() != qtp2.getStorageTypeMin() ||
        qtp1.getStorageTypeMax() != qtp2.getStorageTypeMax())
      return false;
  }
  auto etp1 = getExpressedTypeOrSelf(tp1);
  auto etp2 = getExpressedTypeOrSelf(tp2);

  // Sparsity: In the most general case, we allow any combination of
  // sparsity/denseness across any combination of operands/results, as well as
  // differences in sparsity encodings for operands and results.
  // Individual ops may introduce additional constraints.
  // No additional code is needed to check this because of how sparsity is
  // currently implemented.

  // Default case: Unless dynamism, quantization and/or sparsity are involved,
  // the types are required to be exactly equal.
  return etp1 == etp2;
}

bool isCompatibleForHloTypeInference(TypeRange tp1, TypeRange tp2) {
  if (tp1.size() != tp2.size()) return false;
  for (auto [lt, rt] : llvm::zip(tp1, tp2))
    if (!isCompatibleForHloTypeInference(lt, rt)) return false;
  return true;
}

LogicalResult deriveShapeFromOperand(
    OpBuilder* builder, Operation* op, Value operand,
    SmallVectorImpl<Value>* reifiedReturnShapes) {
  auto shapedTy = operand.getType().dyn_cast<ShapedType>();
  if (!shapedTy) {
    op->emitOpError() << "operand is not a shaped type";
    return failure();
  }
  reifiedReturnShapes->assign(
      {builder->create<shape::ShapeOfOp>(op->getLoc(), operand)});
  return success();
}

TensorType getSameShapeTensorType(TensorType tensorType, Type elementType) {
  if (auto rankedTensorTy = tensorType.dyn_cast<RankedTensorType>()) {
    return RankedTensorType::get(rankedTensorTy.getShape(), elementType,
                                 rankedTensorTy.getEncoding());
  }
  if (auto unrankedTensorTy = tensorType.dyn_cast<UnrankedTensorType>()) {
    return UnrankedTensorType::get(elementType);
  }
  llvm_unreachable("unhandled type");
}

// createRealType takes a tensor type that may have complex elements and
// returns a type that maintains the shape, but with real numeric data types.
//   Ex: tensor<4xcomplex<f32>>  -->  tensor<4xf32>
Type createRealType(TensorType type) {
  auto elementTy = type.getElementType();
  if (auto complexTy = elementTy.dyn_cast<ComplexType>()) {
    elementTy = complexTy.getElementType();
  }
  return hlo::getSameShapeTensorType(type, elementTy);
}

LogicalResult verifyBounds(ArrayRef<int64_t> bounds, RankedTensorType type,
                           function_ref<InFlightDiagnostic()> emitError) {
  int64_t boundsLen = bounds.size();
  int64_t rank = type.getRank();
  if (boundsLen != rank) {
    return emitError() << "Bounds length is " << boundsLen
                       << ", expected to be equal to rank(" << rank
                       << ") of the tensor";
  }

  for (int64_t dim = 0; dim < rank; ++dim) {
    int64_t bound = bounds[dim];
    int64_t dimSize = type.getDimSize(dim);
    if (bound != ShapedType::kDynamic && dimSize != ShapedType::kDynamic) {
      return emitError() << "Static dimension " << dim
                         << " cannot have a bound, use ShapedType::kDynamic to "
                            "indicate a missing bound";
    }
  }

  return success();
}

ArrayRef<int64_t> encodingToBounds(Attribute encoding) {
  if (auto boundedAttr = encoding.dyn_cast_or_null<BoundedAttrInterface>())
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
  int64_t size = ShapedType::kDynamic;
  int64_t bound = ShapedType::kDynamic;

  if (isLeftStaticDim && isRightStaticDim) {
    size = leftSize + rightSize;
  } else {
    int64_t leftSizeOrBound = isLeftStaticDim ? leftSize : leftBound;
    int64_t rightSizeOrBound = isRightStaticDim ? rightSize : rightBound;
    if (!isDynamicDimSize(leftSizeOrBound) &&
        !isDynamicDimSize(rightSizeOrBound))
      bound = leftSizeOrBound + rightSizeOrBound;
  }
  return {size, bound};
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
FailureOr<std::pair<int64_t, int64_t>> inferMergedDimAndBound(
    Optional<Location> location, int64_t dim, int64_t leftSize,
    int64_t rightSize, int64_t leftBound, int64_t rightBound) {
  bool isLeftStaticDim = !isDynamicDimSize(leftSize);
  bool isRightStaticDim = !isDynamicDimSize(rightSize);
  bool isLeftStaticBound = !isDynamicDimSize(leftBound);
  bool isRightStaticBound = !isDynamicDimSize(rightBound);
  int64_t size = ShapedType::kDynamic;
  int64_t bound = ShapedType::kDynamic;

  if (isLeftStaticDim || isRightStaticDim) {
    if (isLeftStaticDim && isRightStaticDim && leftSize != rightSize)
      return emitOptionalError(location, "Mismatched dimension sizes ",
                               leftSize, " and ", rightSize, " in dimension ",
                               dim);
    size = isLeftStaticDim ? leftSize : rightSize;
    if (isLeftStaticBound || isRightStaticBound) {
      int64_t check_bound = isLeftStaticBound ? leftBound : rightBound;
      if (size > check_bound)
        return emitOptionalError(location, "Mismatched dimension size ", size,
                                 " and bound ", check_bound, " in dimension ",
                                 dim);
    }
  } else {
    if (isLeftStaticBound && isRightStaticBound)
      bound = std::min(leftBound, rightBound);
    else
      bound = isLeftStaticBound ? leftBound : rightBound;
  }
  return std::make_pair(size, bound);
}

// TODO(zhouxin) Refactor to better handle errors and return single type
LogicalResult inferMostSpecificType(
    Optional<Location> location, TypeRange inputTypes,
    SmallVectorImpl<Type>& inferredReturnTypes) {
  SmallVector<RankedTensorType> rankedTypes;
  for (auto inputType : inputTypes)
    if (auto rankedType = inputType.dyn_cast<RankedTensorType>())
      rankedTypes.push_back(rankedType);
  if (rankedTypes.empty()) {
    inferredReturnTypes.push_back(inputTypes[0]);
    return success();
  }

  auto rank = rankedTypes[0].getRank();
  SmallVector<int64_t> inferredSizes(rank, ShapedType::kDynamic);
  SmallVector<int64_t> inferredBounds(rank, ShapedType::kDynamic);
  bool anyInputHaveBounds = false;

  for (const auto& it : llvm::enumerate(rankedTypes)) {
    RankedTensorType rankedType = it.value();
    ArrayRef<int64_t> bounds = encodingToBounds(rankedType.getEncoding());
    if (!bounds.empty()) anyInputHaveBounds = true;

    for (int dim = 0; dim < rank; ++dim) {
      std::pair<int64_t, int64_t> inferredDimAndBound;
      int64_t leftSize = inferredSizes[dim];
      int64_t rightSize = rankedType.getShape()[dim];
      int64_t leftBound = inferredBounds[dim];
      int64_t rightBound = bounds.empty() ? ShapedType::kDynamic : bounds[dim];

      auto inferredDimAndBoundOrErr = inferMergedDimAndBound(
          location, dim, leftSize, rightSize, leftBound, rightBound);
      if (failed(inferredDimAndBoundOrErr)) return failure();
      inferredDimAndBound = *inferredDimAndBoundOrErr;
      inferredSizes[dim] = inferredDimAndBound.first;
      inferredBounds[dim] = inferredDimAndBound.second;
    }
  }

  inferredReturnTypes.push_back(RankedTensorType::get(
      inferredSizes, rankedTypes[0].getElementType(),
      boundsToEncoding(
          rankedTypes[0].getEncoding(),
          // Empty array as argument is an indicator to boundsToEncoding() that
          // there are no bounds at all in inputs, thus sparsity attributes will
          // be included in the return type
          anyInputHaveBounds ? inferredBounds : llvm::ArrayRef<int64_t>({}))));
  return success();
}

LogicalResult inferMostSpecificTypeComponents(
    Optional<Location> location, TypeRange inputTypes,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {
  SmallVector<Type> inferredReturnTypes;
  if (failed(hlo::inferMostSpecificType(location, inputTypes,
                                        inferredReturnTypes))) {
    return failure();
  }
  for (auto inferredReturnType : inferredReturnTypes) {
    inferredReturnShapes.emplace_back(inferredReturnType.cast<ShapedType>());
  }
  return success();
}

}  // namespace hlo
}  // namespace mlir
