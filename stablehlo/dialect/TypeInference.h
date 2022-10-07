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

#ifndef STABLEHLO_DIALECT_STABLEHLO_TYPE_INFERENCE_H
#define STABLEHLO_DIALECT_STABLEHLO_TYPE_INFERENCE_H

#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Quant/QuantTypes.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/TensorEncoding.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LogicalResult.h"
#include "stablehlo/dialect/Base.h"

namespace mlir {
namespace stablehlo {

// Check if the dimension size is dynamic.
inline static bool isDynamicDimSize(int64_t val) {
  return val == ShapedType::kDynamicSize;
}

// TODO(https://github.com/openxla/stablehlo/issues/270)
// Remove the util functions below when all the shape functions are moved here.

bool compatibleShapeAndElementType(Type type1, Type type2,
                                   bool ignoreFpPrecision = false);

FailureOr<SmallVector<int64_t>> convert1DAttribute(
    Optional<DenseIntElementsAttr> optionalAttr, Optional<Location> loc,
    StringRef attrName);

FailureOr<SmallVector<std::pair<int64_t, int64_t>>> convertPaddingAttribute(
    Optional<DenseIntElementsAttr> optionalAttr, Optional<Location> loc);

// WindowDimension described how the kernel window moves across the base area
// in a particular dimension.
// Describes the windowing in an operation such as convolution.
// The window is moved across a base area and for each position of the
// window a computation is performed. The field below describes the
// window and the movement of the window across a base area.
struct WindowDimension {
  int64_t size = 0;
  int64_t stride = 1;
  int64_t paddingLow = 0;
  int64_t paddingHigh = 0;
  int64_t windowDilation = 1;
  int64_t baseDilation = 1;
  bool windowReversal = false;
};

FailureOr<SmallVector<WindowDimension>>
verifyWindowAttributesAndInferWindowDimensions(
    ArrayRef<int64_t> windowDimensions, ArrayRef<int64_t> windowStrides,
    ArrayRef<std::pair<int64_t, int64_t>> padding,
    ArrayRef<int64_t> lhsDilation, ArrayRef<int64_t> rhsDilation,
    Optional<Location> loc);

SmallVector<int64_t> inferWindowOutputShape(
    const ArrayRef<int64_t> baseShape, const ArrayRef<WindowDimension> window);

unsigned potentiallyComplexBitwidth(Type type);

//===----------------------------------------------------------------------===//
// Shape functions for ops.
//===----------------------------------------------------------------------===//

LogicalResult inferBatchNormGradOp(
    Value operand, uint64_t featureIndex,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes);

LogicalResult inferBatchNormInferenceOp(
    Value operand, SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes);

LogicalResult inferBatchNormTrainingOp(
    Value operand, uint64_t featureIndex,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes);

LogicalResult inferIfOp(
    Optional<Location> location, RegionRange branches,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes);

LogicalResult inferCaseOp(
    Optional<Location> location, RegionRange branches,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes);

LogicalResult inferDotGeneralOp(
    Optional<Location> location, Value lhs, Value rhs,
    ArrayRef<int64_t> lhsBatchingDims, ArrayRef<int64_t> rhsBatchingDims,
    ArrayRef<int64_t> lhsContractingDims, ArrayRef<int64_t> rhsContractingDims,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes);

LogicalResult inferMapOp(
    Optional<Location> location, ValueRange inputs,
    DenseIntElementsAttr dimensions, Region& computation,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes);

LogicalResult verifyReducerShape(
    Optional<Location> loc, Block& block, ArrayRef<TensorType> inputArgTypes,
    ArrayRef<TensorType> initValueTypes, int64_t numInputs,
    ArrayRef<int64_t> allowedDimensions, bool allInputsUnranked,
    SmallVectorImpl<TensorType>& accumulatorSubShapes);

LogicalResult inferReduceOp(
    Optional<Location> location, ValueRange inputs, ValueRange initValues,
    DenseIntElementsAttr dimensions, Region& body,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes);

LogicalResult inferReduceWindowOp(
    Optional<Location> location, ValueRange inputs, ValueRange initValues,
    DenseIntElementsAttr windowDimensions,
    Optional<DenseIntElementsAttr> windowStrides,
    Optional<DenseIntElementsAttr> baseDilations,
    Optional<DenseIntElementsAttr> windowDilations,
    Optional<DenseIntElementsAttr> padding, Region& body,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes);

LogicalResult inferSortOp(
    Optional<Location> location, ValueRange inputs, uint64_t dimension,
    Region& comparator,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes);

LogicalResult inferTriangularSolveOp(
    Optional<Location> location, Value a, Value b, bool leftSide,
    bool is_transpose_a_invalid,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes);

LogicalResult inferWhileOp(
    Optional<Location> location, ValueRange operand, Region& cond, Region& body,
    TypeRange condReturnTypes, TypeRange bodyReturnTypes,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes);

}  // end namespace stablehlo
}  // end namespace mlir

#endif  // STABLEHLO_DIALECT_STABLEHLO_TYPE_INFERENCE_H
