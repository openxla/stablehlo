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

LogicalResult verifyCompatibleShapeWithBounds(Type type1, Type type2) {
  if (failed(verifyCompatibleShape(type1, type2))) return failure();

  // Verify shapes against bounds
  auto isCompatible = [](ArrayRef<int64_t> shape,
                         BoundedAttrInterface boundedAttr) {
    if (shape.empty() || !boundedAttr) return true;
    auto bounds = boundedAttr.getBounds();
    for (auto [dim_size, bound] : llvm::zip(shape, bounds))  // NOLINT
      if (bound != ShapedType::kDynamicSize && bound < dim_size) return false;
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
}  // namespace

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

bool isCompatibleForHloTypeInference(TypeRange l, TypeRange r) {
  if (l.size() != r.size()) return false;
  for (auto [lt, rt] : llvm::zip(l, r))
    if (!isCompatibleForHloTypeInference(lt, rt)) return false;
  return true;
}

// Cases of infer return shape with bounds (lhs and rhs are commutative):
//       Dim of lhs     Dim of rhs      Infer
//  c0:  3              3               3
//  c1:  3              ?               3
//  c2:  3              ?, bound=4      3
//  c3:  3              ?, bound=2      Error out
//  c4:  ?              ?               ?
//  c5:  ?              ?, bound=3      ?, bound=3
//  c6:  ?, bound=3     ?, bound=3      ?, bound=3
//  c7:  ?, bound=3     ?, bound=4      ?, bound=3
// This method generalizes it to multiple inputs: 1) get the static input dims
// (if any) as infer dim, and 2) get min of input bounds as infer bound
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
  BoundedDialectInterface* dialect = nullptr;
  SmallVector<int64_t> inferredDimSizes(rank, ShapedType::kDynamicSize);
  SmallVector<int64_t> inferredBounds(rank, ShapedType::kDynamicSize);
  for (auto rankedType : rankedTypes) {
    SmallVector<int64_t> bounds;
    if (auto boundedAttr =
            rankedType.getEncoding().dyn_cast_or_null<BoundedAttrInterface>()) {
      dialect = cast<BoundedDialectInterface>(&boundedAttr.getDialect());
      bounds = llvm::to_vector<4>(boundedAttr.getBounds());
    } else if (rankedType.getEncoding()) {
      // TODO(zhouxin) infer sparsity encoding after b/238903065 is fixed.
      inferredReturnTypes.push_back(inputTypes[0]);
      return success();
    }

    for (int dim = 0; dim < rank; ++dim) {
      // Dimensions
      auto dimSize = rankedType.getShape()[dim];
      if (inferredDimSizes[dim] != ShapedType::kDynamicSize &&
          dimSize != ShapedType::kDynamicSize &&
          inferredDimSizes[dim] != dimSize)
        return emitOptionalError(location, "Mismatch dimension size ",
                                 inferredDimSizes[dim], " and ", dimSize,
                                 " in dimension ", dim);
      if (inferredDimSizes[dim] == ShapedType::kDynamicSize)
        inferredDimSizes[dim] = dimSize;

      // Bounds
      if (!bounds.empty() && bounds[dim] != ShapedType::kDynamicSize) {
        if (inferredBounds[dim] == ShapedType::kDynamicSize) {
          inferredBounds[dim] = bounds[dim];
        } else {
          inferredBounds[dim] = std::min(inferredBounds[dim], bounds[dim]);
        }
      }
      // Error out case that the inferred bound is smaller than inferred dim
      if (inferredBounds[dim] != ShapedType::kDynamicSize &&
          inferredBounds[dim] < inferredDimSizes[dim])
        return emitOptionalError(location,
                                 "bound must not be less than static "
                                 "dimension size but has bound ",
                                 inferredBounds[dim], " vs static size ",
                                 inferredDimSizes[dim], " in dimension ", dim);
      if (inferredDimSizes[dim] != ShapedType::kDynamicSize)
        inferredBounds[dim] = ShapedType::kDynamicSize;
    }
  }

  Attribute encoding = nullptr;
  if (llvm::any_of(inferredBounds,
                   [](auto el) { return el != ShapedType::kDynamicSize; })) {
    encoding = dialect->createBoundedAttr(inferredBounds);
  }
  inferredReturnTypes.push_back(RankedTensorType::get(
      inferredDimSizes, rankedTypes[0].getElementType(), encoding));

  return success();
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

// TODO(hinsu): Add verification for bounds that it has the same size as rank
// of the tensor and static dimensions don't have bounds.
LogicalResult verifyBounds(ArrayRef<int64_t> bounds, ShapedType type,
                           function_ref<InFlightDiagnostic()> emitError) {
  return success();
}

}  // namespace hlo
}  // namespace mlir
