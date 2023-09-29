/* Copyright 2023 The StableHLO Authors.

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

#ifndef STABLEHLO_REFERENCE_OPS_H
#define STABLEHLO_REFERENCE_OPS_H

#include "mlir/IR/BuiltinAttributes.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/reference/Axes.h"
#include "stablehlo/reference/InterpreterValue.h"
#include "stablehlo/reference/Process.h"
#include "stablehlo/reference/ProcessGrid.h"
#include "stablehlo/reference/Scope.h"
#include "stablehlo/reference/Tensor.h"
#include "stablehlo/reference/Token.h"

namespace mlir {
namespace stablehlo {

// Evaluators for StableHLO ops.
Tensor evalAbsOp(const Tensor &operand, ShapedType resultType);
Tensor evalAddOp(const Tensor &lhs, const Tensor &rhs, ShapedType resultType);
Token evalAfterAllOp(ArrayRef<Token> inputs, MLIRContext *context);
Tensor evalAllGatherOp(const Tensor &operand, int64_t allGatherDim,
                       SmallVector<SmallVector<uint32_t>> replicaGroups,
                       ChannelId channelId, bool useGlobalDeviceIds,
                       Process *process, ShapedType resultType);
Tensor evalAllReduceOp(const Tensor &operand,
                       SmallVector<SmallVector<uint32_t>> replicaGroups,
                       ChannelId channelId, bool useGlobalDeviceIds,
                       Region &computation, Process *process, Scope &scope,
                       ShapedType resultType);
Tensor evalAllToAllOp(const Tensor &operand, Axis splitDimension,
                      Axis concatDimension, int64_t splitCount,
                      SmallVector<SmallVector<uint32_t>> replicaGroups,
                      ChannelId channelId, Process *process,
                      ShapedType resultType);
Tensor evalAndOp(const Tensor &lhs, const Tensor &rhs, ShapedType resultType);
Tensor evalAtan2Op(const Tensor &lhs, const Tensor &rhs, ShapedType resultType);
Tensor evalBitcastConvertOp(const Tensor &operand, ShapedType resultType);
Tensor evalBroadcastInDimOp(const Tensor &operand,
                            const Axes &broadcastDimensions,
                            ShapedType resultType);
SmallVector<InterpreterValue> evalCaseOp(const Tensor &index,
                                         RegionRange branches, Process *process,
                                         Scope &scope);
Tensor evalCbrtOp(const Tensor &operand, ShapedType resultType);
Tensor evalCeilOp(const Tensor &operand, ShapedType resultType);
Tensor evalClampOp(const Tensor &min, const Tensor &operand, const Tensor &max,
                   ShapedType resultType);
Tensor evalClzOp(const Tensor &operand, ShapedType resultType);
Tensor evalCollectivePermuteOp(
    const Tensor &operand, SmallVector<SmallVector<uint32_t>> sourceTargetPairs,
    ChannelId channelId, Process *process);
Tensor evalCompareOp(const Tensor &lhs, const Tensor &rhs,
                     ComparisonDirection comparisonDirection,
                     ShapedType resultType);
Tensor evalComplexOp(const Tensor &lhs, const Tensor &rhs,
                     ShapedType resultType);
Tensor evalConcatenateOp(ArrayRef<Tensor> inputs, Axis dimension,
                         ShapedType resultType);
Tensor evalConstantOp(ElementsAttr value);
Tensor evalConvertOp(const Tensor &operand, ShapedType resultType);
Tensor evalCosineOp(const Tensor &operand, ShapedType resultType);
Tensor evalDivideOp(const Tensor &lhs, const Tensor &rhs,
                    ShapedType resultType);
Tensor evalDotGeneralOp(const Tensor &lhs, const Tensor &rhs,
                        const Axes &lhsBatchingDimensions,
                        const Axes &rhsBatchingDimensions,
                        const Axes &lhsContractingDimensions,
                        const Axes &rhsContractingDimensions,
                        ShapedType resultType);
Tensor evalDynamicSliceOp(const Tensor &operand, ArrayRef<Tensor> startIndices,
                          const Sizes &sliceSizes, ShapedType resultType);
Tensor evalDynamicUpdateSliceOp(const Tensor &operand, const Tensor &update,
                                ArrayRef<Tensor> startIndices,
                                ShapedType resultType);
Tensor evalExpm1Op(const Tensor &operand, ShapedType resultType);
Tensor evalExponentialOp(const Tensor &operand, ShapedType resultType);
Tensor evalFloorOp(const Tensor &operand, ShapedType resultType);
Tensor evalGatherOp(const Tensor &operand, const Tensor &startIndices,
                    const Axes &offsetDims, const Axes &collapsedSliceDims,
                    const Axes &startIndexMap, Axis indexVectorDim,
                    const Sizes &sliceSizes, bool indicesAreSorted,
                    ShapedType resultType);
Tensor evalGetDimensionSizeOp(const Tensor &operand, Axis dimension,
                              ShapedType resultType);
InterpreterValue evalGetTupleElementOp(const Tuple &operand, int32_t index);
SmallVector<InterpreterValue> evalIfOp(const Tensor &pred, Region &trueBranch,
                                       Region &falseBranch, Process *process,
                                       Scope &scope);
Tensor evalImagOp(const Tensor &operand, ShapedType resultType);
SmallVector<InterpreterValue> evalInfeedOp(Token token, Process *process,
                                           Region &region, Scope &scope);
Tensor evalIotaOp(Axis iotaDimension, ShapedType resultType);
Tensor evalIsFiniteOp(const Tensor &operand, ShapedType resultType);
Tensor evalLog1pOp(const Tensor &operand, ShapedType resultType);
Tensor evalLogOp(const Tensor &operand, ShapedType resultType);
Tensor evalLogisticOp(const Tensor &operand, ShapedType resultType);
Tensor evalMapOp(ArrayRef<Tensor> inputs, Region &computation, Process *process,
                 Scope &scope, ShapedType resultType);
Tensor evalMaxOp(const Tensor &lhs, const Tensor &rhs, ShapedType resultType);
Tensor evalMinOp(const Tensor &lhs, const Tensor &rhs, ShapedType resultType);
Tensor evalMultiplyOp(const Tensor &lhs, const Tensor &rhs,
                      ShapedType resultType);
Tensor evalNegOp(const Tensor &operand, ShapedType resultType);
Tensor evalNotOp(const Tensor &operand, ShapedType resultType);
SmallVector<InterpreterValue> evalOptimizationBarrierOp(
    ArrayRef<InterpreterValue> operand);
Tensor evalOrOp(const Tensor &lhs, const Tensor &rhs, ShapedType resultType);
Token evalOutfeedOp(ArrayRef<Tensor> inputs, Token token, Process *process);
Tensor evalPadOp(const Tensor &operand, const Tensor &paddingValue,
                 const Sizes &edgePaddingLow, const Sizes &interiorPadding,
                 ShapedType resultType);
Tensor evalPartitionIdOp(Process *process, MLIRContext *context);
Tensor evalPopulationCountOp(const Tensor &operand, ShapedType resultType);
Tensor evalPowerOp(const Tensor &lhs, const Tensor &rhs, ShapedType resultType);
Tensor evalRealOp(const Tensor &operand, ShapedType resultType);
SmallVector<InterpreterValue> evalRecvOp(Token token, ChannelId channelId,
                                         Process *process);
SmallVector<Tensor> evalReduceOp(ArrayRef<Tensor> inputs,
                                 ArrayRef<Tensor> initValues,
                                 const Axes &dimensions, Region &body,
                                 Process *process, Scope &scope,
                                 ArrayRef<ShapedType> resultTypes);
Tensor evalReducePrecisionOp(const Tensor &operand, int32_t exponentBits,
                             int32_t mantissaBits, ShapedType resultType);
Tensor evalReduceScatterOp(const Tensor &operand, int64_t scatterDimension,
                           SmallVector<SmallVector<uint32_t>> replicaGroups,
                           ChannelId channelId, bool useGlobalDeviceIds,
                           Region &region, Process *process, Scope &scope,
                           ShapedType returnType);
SmallVector<Tensor> evalReduceWindowOp(
    ArrayRef<Tensor> inputs, ArrayRef<Tensor> initValues,
    const Sizes &windowDimensions, const Sizes &windowStrides,
    const Sizes &baseDilations, const Sizes &windowDilations,
    const Sizes &paddingLow, const Sizes &paddingHigh, Region &body,
    Process *process, Scope &scope, ArrayRef<ShapedType> resultTypes);
Tensor evalRemOp(const Tensor &lhs, const Tensor &rhs, ShapedType resultType);
Tensor evalReplicaIdOp(Process *process, MLIRContext *context);
Tensor evalReshapeOp(const Tensor &operand, ShapedType resultType);
Tensor evalReverseOp(const Tensor &operand, const Axes &dimensions,
                     ShapedType resultType);
Tensor evalRoundOp(const Tensor &operand, ShapedType resultType);
Tensor evalRoundNearestEvenOp(const Tensor &operand, ShapedType resultType);
Tensor evalRsqrtOp(const Tensor &operand, ShapedType resultType);
SmallVector<Tensor> evalScatterOp(
    ArrayRef<Tensor> inputs, const Tensor &scatterIndices,
    ArrayRef<Tensor> updates, const Axes &updateWindowDims,
    const Axes &insertedWindowDims, const Axes &scatterDimsToOperandDims,
    Axis indexVectorDim, Region &updateComputation, Process *process,
    Scope &scope, ArrayRef<ShapedType> resultTypes);
Tensor evalSelectOp(const Tensor &pred, const Tensor &onTrue,
                    const Tensor &onFalse, ShapedType resultType);
Tensor evalSelectAndScatterOp(const Tensor &operand, const Tensor &source,
                              const Tensor &initValue,
                              const Sizes &windowDimensions,
                              const Sizes &windowStrides,
                              const Sizes &paddingLow, Region &select,
                              Region &scatter, Process *process, Scope &scope,
                              ShapedType resultType);
Token evalSendOp(ArrayRef<Tensor> inputs, Token token, ChannelId channelId,
                 Process *process);
Tensor evalShiftLeftOp(const Tensor &lhs, const Tensor &rhs,
                       ShapedType resultType);
Tensor evalShiftRightArithmeticOp(const Tensor &lhs, const Tensor &rhs,
                                  ShapedType resultType);
Tensor evalShiftRightLogicalOp(const Tensor &lhs, const Tensor &rhs,
                               ShapedType resultType);
Tensor evalSignOp(const Tensor &operand, ShapedType resultType);
Tensor evalSineOp(const Tensor &operand, ShapedType resultType);
Tensor evalSliceOp(const Tensor &operand, const Sizes &startIndices,
                   const Sizes &strides, ShapedType resultType);
SmallVector<Tensor> evalSortOp(ArrayRef<Tensor> inputs, Axis dimension,
                               bool isStable, Region &comparator,
                               Process *process, Scope &scope);
Tensor evalSqrtOp(const Tensor &operand, ShapedType resultType);
Tensor evalSubtractOp(const Tensor &lhs, const Tensor &rhs,
                      ShapedType resultType);
Tensor evalTanhOp(const Tensor &operand, ShapedType resultType);
Tensor evalTransposeOp(const Tensor &operand, const Axes &permutation,
                       ShapedType resultType);
Tuple evalTupleOp(ArrayRef<InterpreterValue> val, TupleType resultType);
SmallVector<InterpreterValue> evalWhileOp(SmallVector<InterpreterValue> operand,
                                          Region &cond, Region &body,
                                          Process *process, Scope &scope);
Tensor evalXorOp(const Tensor &lhs, const Tensor &rhs, ShapedType resultType);

/// Evaluates an mlir::Region `region` using the runtime values `args`
/// corresponding to the arguments of the entry block of the region.
/// Interprets the operations within the entry block and returns the runtime
/// values for the terminator's arguments. The optional callback `fallback` is
/// used for evaluating ops which are not supported by the interpreter.
/// Assumes that `region` has only one block.
SmallVector<InterpreterValue> eval(
    Region &region, ArrayRef<InterpreterValue> args, Process *process = nullptr,
    Scope *parent = nullptr,
    function_ref<llvm::Error(Operation &, Process *, Scope &)> fallback =
        nullptr);

}  // namespace stablehlo
}  // namespace mlir

#endif  // STABLEHLO_REFERENCE_OPS_H
