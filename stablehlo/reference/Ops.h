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

#ifndef STABLEHLO_REFERENCE_OPS_H
#define STABLEHLO_REFERENCE_OPS_H

#include "mlir/IR/BuiltinAttributes.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/reference/Axes.h"
#include "stablehlo/reference/Scope.h"
#include "stablehlo/reference/Sizes.h"
#include "stablehlo/reference/Tensor.h"

namespace mlir {
namespace stablehlo {

// Evaluators for StableHLO ops.
Tensor evalAbsOp(const Tensor &operand, ShapedType resultType);
Tensor evalAddOp(const Tensor &lhs, const Tensor &rhs, ShapedType resultType);
Tensor evalAndOp(const Tensor &lhs, const Tensor &rhs, ShapedType resultType);
Tensor evalAtan2Op(const Tensor &lhs, const Tensor &rhs, ShapedType resultType);
Tensor evalBroadcastInDimOp(const Tensor &operand,
                            const Axes &broadcastDimensions,
                            ShapedType resultType);
SmallVector<Tensor> evalCaseOp(const Tensor &index, RegionRange branches,
                               Scope &scope);
Tensor evalCbrtOp(const Tensor &operand, ShapedType resultType);
Tensor evalCeilOp(const Tensor &operand, ShapedType resultType);
Tensor evalClampOp(const Tensor &min, const Tensor &operand, const Tensor &max,
                   ShapedType resultType);
Tensor evalClzOp(const Tensor &operand, ShapedType resultType);
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
SmallVector<Tensor> evalIfOp(const Tensor &pred, Region &trueBranch,
                             Region &falseBranch, Scope &scope);
Tensor evalImagOp(const Tensor &operand, ShapedType resultType);
Tensor evalIotaOp(Axis iotaDimension, ShapedType resultType);
Tensor evalIsFiniteOp(const Tensor &operand, ShapedType resultType);
Tensor evalLog1pOp(const Tensor &operand, ShapedType resultType);
Tensor evalLogOp(const Tensor &operand, ShapedType resultType);
Tensor evalLogisticOp(const Tensor &operand, ShapedType resultType);
Tensor evalMapOp(ArrayRef<Tensor> inputs, Region &computation, Scope &scope,
                 ShapedType resultType);
Tensor evalMaxOp(const Tensor &lhs, const Tensor &rhs, ShapedType resultType);
Tensor evalMinOp(const Tensor &lhs, const Tensor &rhs, ShapedType resultType);
Tensor evalMultiplyOp(const Tensor &lhs, const Tensor &rhs,
                      ShapedType resultType);
Tensor evalNegOp(const Tensor &operand, ShapedType resultType);
Tensor evalNotOp(const Tensor &operand, ShapedType resultType);
Tensor evalOrOp(const Tensor &lhs, const Tensor &rhs, ShapedType resultType);
Tensor evalPadOp(const Tensor &operand, const Tensor &paddingValue,
                 const Sizes &edgePaddingLow, const Sizes &interiorPadding,
                 ShapedType resultType);
Tensor evalPopulationCountOp(const Tensor &operand, ShapedType resultType);
Tensor evalPowerOp(const Tensor &lhs, const Tensor &rhs, ShapedType resultType);
Tensor evalRealOp(const Tensor &operand, ShapedType resultType);
SmallVector<Tensor> evalReduceOp(ArrayRef<Tensor> inputs,
                                 ArrayRef<Tensor> initValues,
                                 const Axes &dimensions, Region &body,
                                 Scope &scope,
                                 ArrayRef<ShapedType> resultTypes);
SmallVector<Tensor> evalReduceWindowOp(
    ArrayRef<Tensor> inputs, ArrayRef<Tensor> initValues,
    const Sizes &windowDimensions, const Sizes &windowStrides,
    const Sizes &baseDilations, const Sizes &windowDilations,
    const Sizes &paddingLow, const Sizes &paddingHigh, Region &body,
    Scope &scope, ArrayRef<ShapedType> resultTypes);
Tensor evalRemOp(const Tensor &lhs, const Tensor &rhs, ShapedType resultType);
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
    Axis indexVectorDim, Region &updateComputation, Scope &scope,
    ArrayRef<ShapedType> resultTypes);
Tensor evalSelectOp(const Tensor &pred, const Tensor &onTrue,
                    const Tensor &onFalse, ShapedType resultType);
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
                               bool isStable, Region &comparator, Scope &scope);
Tensor evalSqrtOp(const Tensor &operand, ShapedType resultType);
Tensor evalSubtractOp(const Tensor &lhs, const Tensor &rhs,
                      ShapedType resultType);
Tensor evalTanhOp(const Tensor &operand, ShapedType resultType);
Tensor evalTransposeOp(const Tensor &operand, const Axes &permutation,
                       ShapedType resultType);
SmallVector<Tensor> evalWhileOp(ArrayRef<Tensor> operand, Region &cond,
                                Region &body, Scope &scope);
Tensor evalXorOp(const Tensor &lhs, const Tensor &rhs, ShapedType resultType);

/// Evaluates an mlir::Region `region` using the runtime values `args`
/// corresponding to the arguments of the entry block of the region.
/// Interprets the operations within the entry block and returns the runtime
/// values for the terminator's arguments. The optional callback `fallback` is
/// used for evaluating ops which are not supported by the interpreter.
/// Assumes that `region` has only one block.
llvm::SmallVector<Tensor> eval(
    Region &region, llvm::ArrayRef<Tensor> args, Scope *parent = nullptr,
    llvm::function_ref<llvm::Error(Operation &, Scope &)> fallback = nullptr);

}  // namespace stablehlo
}  // namespace mlir

#endif  // STABLEHLO_REFERENCE_OPS_H
