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

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace hlo {

//===----------------------------------------------------------------------===//
// Utilities for shape functions
//===----------------------------------------------------------------------===//
// TODO(#270): Remove them when all shape functions are moved to this file.

bool compatibleShapeAndElementType(Type type1, Type type2,
                                   bool ignoreFpPrecision = false);

FailureOr<SmallVector<int64_t>> convert1DAttribute(
    Optional<DenseIntElementsAttr> optionalAttr, Optional<Location> loc,
    StringRef attrName);

FailureOr<SmallVector<std::pair<int64_t, int64_t>>> convertPaddingAttribute(
    Optional<DenseIntElementsAttr> optionalAttr, Optional<Location> loc);

// Convert a 1D dense bool attribute to a list of values.
FailureOr<SmallVector<bool>> convertWindowReversalAttribute(
    Optional<DenseElementsAttr> optionalAttr, Optional<Location> loc,
    StringRef attrName);

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
    ArrayRef<bool> windowReversal, Optional<Location> loc);

SmallVector<int64_t> inferWindowOutputShape(
    const ArrayRef<int64_t> baseShape, const ArrayRef<WindowDimension> window);

unsigned potentiallyComplexBitwidth(Type type);

LogicalResult verifyReducerShape(Optional<Location> loc, Block& block,
                                 ArrayRef<TensorType> inputArgTypes,
                                 ArrayRef<TensorType> initValueTypes,
                                 int64_t numInputs,
                                 ArrayRef<int64_t> allowedDimensions,
                                 bool allInputsUnranked);

// Verifies replica groups attached to collective communication operations.
// P1. 'replicaGroups' must be a 2-D tensor.
// P2. replicaGroups' cannot be empty.
// P3. If `allGroupsMustHaveSameSize` is true, then each group is of the same
//     size.
// P4. All values in `replica_groups` are unique and covers all the values in
//     the interval [0, N-1], where N is the total number of replica ids.
// P5. replica group size must be equal to 'expectedGroupSize'.
LogicalResult verifyReplicaGroups(Optional<Location> location,
                                  DenseIntElementsAttr replicaGroups,
                                  bool allGroupsMustHaveSameSize,
                                  bool useGlobalDeviceIds,
                                  Optional<size_t> expectedGroupSize);

//===----------------------------------------------------------------------===//
// Shape functions for ops.
//===----------------------------------------------------------------------===//
// These functions have been moved out of StablehloOps.cpp in order to be
// shared with the MHLO dialect.
// Because of that, they cannot use any definitions in the StableHLO dialect
// (definitions in Base are fine, because they are shared with MHLO).
// As a result, these definitions (e.g. StableHLO ops and attributes) are
// decomposed into smaller pieces which are passed as individual parameters.
// These parameters have the same names as in the ODS and come in the same
// order in which they are declared in the ODS.

LogicalResult inferAbsOp(Optional<Location>, Value operand,
                         SmallVectorImpl<Type>& inferredReturnTypes);

LogicalResult inferAfterAllOp(Dialect* dialect, Optional<Location> location,
                              SmallVectorImpl<Type>& inferredReturnTypes);

LogicalResult inferAllToAllOp(
    Optional<Location> location, Value operand, int64_t splitDimension,
    int64_t concatDimension, int64_t splitCount,
    DenseIntElementsAttr replicaGroups,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes);

LogicalResult inferBatchNormGradOp(
    Optional<Location> location, Value operand, Value scale,
    int64_t featureIndex,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes);

LogicalResult inferBatchNormInferenceOp(
    Optional<Location> location, Value operand, Value scale,
    int64_t featureIndex,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes);

LogicalResult inferBatchNormTrainingOp(
    Optional<Location> location, Value operand, Value scale,
    int64_t featureIndex,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes);

LogicalResult inferBroadcastOp(
    Optional<Location> location, Value operand,
    DenseIntElementsAttr broadcastSizes,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes);

LogicalResult inferCaseOp(Optional<Location> location, RegionRange branches,
                          SmallVectorImpl<Type>& inferredReturnTypes);

LogicalResult inferCholeskyOp(
    Optional<Location> location, Value a,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes);

LogicalResult inferClampOp(
    Optional<Location> location, Value min, Value operand, Value max,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes);

LogicalResult inferComplexOp(Optional<Location> location, Value lhs,
                             SmallVectorImpl<Type>& inferredReturnTypes);

LogicalResult inferConcatenateOp(Optional<Location> location, ValueRange inputs,
                                 int64_t dimension,
                                 SmallVectorImpl<Type>& inferredReturnTypes);

LogicalResult inferConstantOp(Optional<Location>, ElementsAttr value,
                              SmallVectorImpl<Type>& inferredReturnTypes);

LogicalResult inferCreateTokenOp(Dialect* dialect, Optional<Location> location,
                                 SmallVectorImpl<Type>& inferredReturnTypes);

LogicalResult inferDynamicSliceOp(
    Optional<Location> location, Value operand, ValueRange startIndices,
    DenseIntElementsAttr sliceSizes,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes);

LogicalResult inferDynamicUpdateSliceOp(
    Optional<Location> location, Value operand, Value update,
    ValueRange startIndices,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes);

LogicalResult inferGetTupleElementOp(
    Optional<Location> location, Value operand, int32_t index,
    SmallVectorImpl<Type>& inferredReturnTypes);

LogicalResult inferIsFiniteOp(MLIRContext* context, Optional<Location>, Value x,
                              SmallVectorImpl<Type>& inferredReturnTypes);

LogicalResult inferGetDimensionSizeOp(
    MLIRContext* context, Optional<Location> location,
    SmallVectorImpl<Type>& inferredReturnTypes);

LogicalResult inferIfOp(Optional<Location> location, RegionRange branches,
                        SmallVectorImpl<Type>& inferredReturnTypes);

LogicalResult inferMapOp(
    Optional<Location> location, ValueRange inputs,
    DenseIntElementsAttr dimensions, Region& computation,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes);

LogicalResult inferPadOp(Optional<Location> location, Value operand,
                         Value paddingValue,
                         DenseIntElementsAttr edgePaddingLow,
                         DenseIntElementsAttr edgePaddingHigh,
                         DenseIntElementsAttr interiorPadding,
                         SmallVectorImpl<Type>& inferredReturnTypes);

LogicalResult inferOptimizationBarrierOp(
    Optional<Location> location, ValueRange operand,
    SmallVectorImpl<Type>& inferredReturnTypes);

LogicalResult inferOutfeedOp(Dialect* dialect, Optional<Location> location,
                             SmallVectorImpl<Type>& inferredReturnTypes);

LogicalResult inferPartitionIdOp(MLIRContext* context,
                                 Optional<Location> location,
                                 SmallVectorImpl<Type>& inferredReturnTypes);

LogicalResult inferRealOp(Optional<Location> location, Value operand,
                          SmallVectorImpl<Type>& inferredReturnTypes);

LogicalResult inferReduceOp(
    Optional<Location> location, ValueRange inputs, ValueRange initValues,
    DenseIntElementsAttr dimensions,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes);

LogicalResult inferReduceWindowOp(
    Optional<Location> location, ValueRange inputs, ValueRange initValues,
    DenseIntElementsAttr windowDimensions,
    Optional<DenseIntElementsAttr> windowStrides,
    Optional<DenseIntElementsAttr> baseDilations,
    Optional<DenseIntElementsAttr> windowDilations,
    Optional<DenseIntElementsAttr> padding,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes);

LogicalResult inferReturnOp(Optional<Location> location,
                            SmallVectorImpl<Type>& inferredReturnTypes);

LogicalResult inferScatterOp(Optional<Location> location, ValueRange inputs,
                             SmallVectorImpl<Type>& inferredReturnTypes);

LogicalResult inferSelectOp(
    Optional<Location> location, Value pred, Value onTrue, Value onFalse,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes);

LogicalResult inferSelectAndScatterOp(
    Value operand, SmallVectorImpl<Type>& inferredReturnTypes);

LogicalResult inferSendOp(Dialect* dialect, Optional<Location> location,
                          SmallVectorImpl<Type>& inferredReturnTypes);

LogicalResult inferSliceOp(Optional<Location> location, Value operand,
                           DenseIntElementsAttr startIndices,
                           DenseIntElementsAttr limitIndices,
                           DenseIntElementsAttr strides,
                           SmallVectorImpl<Type>& inferredReturnTypes);

LogicalResult inferSortOp(
    Optional<Location> location, ValueRange inputs,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes);

LogicalResult inferTransposeOp(Optional<Location> loc, Value operand,
                               DenseIntElementsAttr permutation,
                               SmallVectorImpl<Type>& inferredReturnTypes);

LogicalResult inferTriangularSolveOp(
    Optional<Location> location, Value a, Value b, bool leftSide,
    bool isTransposeAInvalid,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes);

LogicalResult inferTupleOp(MLIRContext* context, Optional<Location> location,
                           ValueRange val,
                           SmallVectorImpl<Type>& inferredReturnTypes);

LogicalResult inferWhileOp(Optional<Location> location, ValueRange operand,
                           SmallVectorImpl<Type>& inferredReturnTypes);

//===----------------------------------------------------------------------===//
// Verifiers for ops.
//===----------------------------------------------------------------------===//

LogicalResult verifyAllReduceOp(Optional<Location> location, Value operand,
                                DenseIntElementsAttr replicaGroups,
                                bool useGlobalDeviceIds, Region& computation);

LogicalResult verifyBitcastConvertOp(Optional<Location> location, Value operand,
                                     Value result);

LogicalResult verifyBroadcastInDimOp(Optional<Location> location, Value operand,
                                     DenseIntElementsAttr broadcastDimensions,
                                     Value result);

LogicalResult verifyCollectivePermuteOp(Optional<Location> location,
                                        DenseIntElementsAttr sourceTargetPairs);

LogicalResult verifyConvolutionOp(
    Optional<Location> location, Value lhs, Value rhs,
    Optional<DenseIntElementsAttr> windowStrides,
    Optional<DenseIntElementsAttr> padding,
    Optional<DenseIntElementsAttr> lhsDilation,
    Optional<DenseIntElementsAttr> rhsDilation,
    Optional<DenseElementsAttr> windowReversal, int64_t inputBatchDimension,
    int64_t inputFeatureDimension, ArrayRef<int64_t> inputSpatialDimensions,
    int64_t kernelInputFeatureDimension, int64_t kernelOutputFeatureDimension,
    ArrayRef<int64_t> kernelSpatialDimensions, int64_t outputBatchDimension,
    int64_t outputFeatureDimension, ArrayRef<int64_t> outputSpatialDimensions,
    int64_t featureGroupCount, int64_t batchGroupCount,
    Optional<ArrayAttr> precisionConfig, Value result);

LogicalResult verifyDotOp(Optional<Location> location, Value lhs, Value rhs,
                          Optional<ArrayAttr> precisionConfig, Value result);

LogicalResult verifyDotGeneralOp(Optional<Location> location, Value lhs,
                                 Value rhs,
                                 ArrayRef<int64_t> lhsBatchingDimensions,
                                 ArrayRef<int64_t> rhsBatchingDimensions,
                                 ArrayRef<int64_t> lhsContractingDimensions,
                                 ArrayRef<int64_t> rhsContractingDimensions,
                                 Optional<ArrayAttr> precisionConfig,
                                 Value result);

LogicalResult verifyDynamicBroadcastInDimOp(
    Optional<Location> location, Value operand, Value outputDimensions,
    DenseIntElementsAttr broadcastDimensions,
    Optional<DenseIntElementsAttr> knownExpandingDimensions,
    Optional<DenseIntElementsAttr> knownNonexpandingDimensions, Value result);

LogicalResult verifyDynamicReshapeOp(Optional<Location> location,
                                     Value outputShape, Value result);

LogicalResult verifyIotaOp(Optional<Location> location, int64_t iotaDimension,
                           Value result);

LogicalResult verifyRealDynamicSliceOp(Optional<Location> location,
                                       Value operand, Value startIndices,
                                       Value limitIndices, Value strides);

LogicalResult verifyReduceOp(Optional<Location> location, ValueRange inputs,
                             ValueRange initValues,
                             DenseIntElementsAttr dimensions, Region& body);

LogicalResult verifyReduceScatterOp(Optional<Location> location, Value operand,
                                    int64_t scatterDimension,
                                    DenseIntElementsAttr replicaGroups,
                                    bool useGlobalDeviceIds,
                                    Region& computation, Value result);

LogicalResult verifyReduceWindowOp(
    Optional<Location> location, ValueRange inputs, ValueRange initValues,
    DenseIntElementsAttr windowDimensions,
    Optional<DenseIntElementsAttr> windowStrides,
    Optional<DenseIntElementsAttr> baseDilations,
    Optional<DenseIntElementsAttr> windowDilations,
    Optional<DenseIntElementsAttr> padding, Region& body);

LogicalResult verifySortOp(Optional<Location> location, ValueRange inputs,
                           int64_t dimension, Region& comparator);

LogicalResult verifyWhileOp(Optional<Location> location, ValueRange operand,
                            Region& cond, Region& body);
}  // end namespace hlo
}  // end namespace mlir

#endif  // STABLEHLO_DIALECT_STABLEHLO_TYPE_INFERENCE_H
