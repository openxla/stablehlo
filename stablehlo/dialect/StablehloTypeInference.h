/* Copyright 2022 The StableHLO Authors.
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

#ifndef ENUMS_AND_ATTRS
#define ENUMS_AND_ATTRS
#include "stablehlo/dialect/StablehloEnums.h.inc"
#define GET_ATTRDEF_CLASSES
#include "stablehlo/dialect/StablehloAttrs.h.inc"
#endif  // ENUMS_AND_ATTRS

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

LogicalResult inferReturnTypeComponentsOfBatchNormGradOp(
    Value operand, uint64_t featureIndex,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes);

LogicalResult inferReturnTypeComponentsOfBatchNormTrainingOp(
    Value operand, uint64_t featureIndex,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes);

LogicalResult inferReturnTypeComponentsOfBatchNormInferenceOp(
    Value operand, SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes);

LogicalResult inferReturnTypeComponentsOfConditionalOp(
    RegionRange branches, Optional<Location> location,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes);

LogicalResult inferReturnTypeComponentsOfDotGeneralOp(
    Value lhs, Value rhs, DotDimensionNumbersAttr dimNumbers,
    Optional<Location> location,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes);

LogicalResult inferReturnTypeComponentsOfMapOp(
    ValueRange inputs, Region& computation, DenseIntElementsAttr dimensions,
    Optional<Location> location,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes);

LogicalResult verifyReducerShape(
    Optional<Location> loc, Block& block, ArrayRef<TensorType> inputArgTypes,
    ArrayRef<TensorType> initValueTypes, int64_t numInputs,
    ArrayRef<int64_t> allowedDimensions, bool allInputsUnranked,
    SmallVectorImpl<TensorType>& accumulatorSubShapes);

LogicalResult inferReturnTypeComponentsOfReduceOp(
    ValueShapeRange operands, DenseIntElementsAttr dimensions, Region& body,
    Optional<Location> location,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes);

LogicalResult inferReturnTypeComponentsOfReduceWindowOp(
    ValueShapeRange operands, DenseIntElementsAttr windowDimensions,
    Optional<DenseIntElementsAttr> padding,
    Optional<DenseIntElementsAttr> windowStrides,
    Optional<DenseIntElementsAttr> baseDilations,
    Optional<DenseIntElementsAttr> windowDilations, Region& body,
    Optional<Location> location,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes);

LogicalResult inferReturnTypeComponentsOfSortOp(
    ValueRange operands, uint64_t dimension, Region& comparator,
    Optional<Location> location,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes);

LogicalResult inferReturnTypeComponentsOfTriangularSolveOp(
    Value a, Value b, bool leftSide, Transpose transposeA,
    Optional<Location> location,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes);

LogicalResult inferReturnTypeComponentsOfWhileOp(
    ValueRange operand, Region& cond, Region& body, Optional<Location> location,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes);

}  // end namespace stablehlo
}  // end namespace mlir

#endif  // STABLEHLO_DIALECT_STABLEHLO_TYPE_INFERENCE_H
