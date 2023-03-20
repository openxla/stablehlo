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
Tensor evalAbsOp(const Tensor &operand, TensorType resultType);
Tensor evalAddOp(const Tensor &lhs, const Tensor &rhs, TensorType resultType);
Tensor evalAndOp(const Tensor &lhs, const Tensor &rhs, TensorType resultType);
Tensor evalBroadcastInDimOp(const Tensor &operand, Axes broadcastDimensions,
                            TensorType resultType);
SmallVector<Tensor> evalCaseOp(const Tensor &index, RegionRange branches,
                               Scope &scope);
Tensor evalCeilOp(const Tensor &operand, TensorType resultType);
Tensor evalClampOp(const Tensor &min, const Tensor &operand, const Tensor &max,
                   TensorType resultType);
Tensor evalCompareOp(const Tensor &lhs, const Tensor &rhs,
                     ComparisonDirection comparisonDirection,
                     std::optional<ComparisonType> compareType,
                     TensorType resultType);
Tensor evalConcatenateOp(ArrayRef<Tensor> inputs, Axis dimension,
                         TensorType resultType);
Tensor evalConstantOp(ElementsAttr value);
Tensor evalConvertOp(const Tensor &operand, TensorType resultType);
Tensor evalCosineOp(const Tensor &operand, TensorType resultType);
Tensor evalDivideOp(const Tensor &lhs, const Tensor &rhs,
                    TensorType resultType);
Tensor evalDynamicSliceOp(const Tensor &operand, ArrayRef<Tensor> startIndices,
                          Sizes sliceSizes, TensorType resultType);
Tensor evalDynamicUpdateSliceOp(const Tensor &operand, const Tensor &update,
                                ArrayRef<Tensor> startIndices,
                                TensorType resultType);
Tensor evalExponentialOp(const Tensor &operand, TensorType resultType);
Tensor evalFloorOp(const Tensor &operand, TensorType resultType);
SmallVector<Tensor> evalIfOp(const Tensor &pred, Region &trueBranch,
                             Region &falseBranch, Scope &scope);
Tensor evalImagOp(const Tensor &operand, TensorType resultType);
Tensor evalIotaOp(Axis iotaDimension, TensorType resultType);
Tensor evalLogOp(const Tensor &operand, TensorType resultType);
Tensor evalLogisticOp(const Tensor &operand, TensorType resultType);
Tensor evalMaxOp(const Tensor &lhs, const Tensor &rhs, TensorType resultType);
Tensor evalMinOp(const Tensor &lhs, const Tensor &rhs, TensorType resultType);
Tensor evalMultiplyOp(const Tensor &lhs, const Tensor &rhs,
                      TensorType resultType);
Tensor evalNegOp(const Tensor &operand, TensorType resultType);
Tensor evalNotOp(const Tensor &operand, TensorType resultType);
Tensor evalOrOp(const Tensor &lhs, const Tensor &rhs, TensorType resultType);
Tensor evalPadOp(const Tensor &operand, const Tensor &paddingValue,
                 Sizes edgePaddingLow, Sizes interiorPadding,
                 TensorType resultType);
Tensor evalPowerOp(const Tensor &lhs, const Tensor &rhs, TensorType resultType);
Tensor evalRealOp(const Tensor &operand, TensorType resultType);
Tensor evalRemOp(const Tensor &lhs, const Tensor &rhs, TensorType resultType);
Tensor evalReshapeOp(const Tensor &operand, TensorType resultType);
Tensor evalReverseOp(const Tensor &operand, Axes dimensions,
                     TensorType resultType);
Tensor evalRsqrtOp(const Tensor &operand, TensorType resultType);
Tensor evalSelectOp(const Tensor &pred, const Tensor &onTrue,
                    const Tensor &onFalse, TensorType resultType);
Tensor evalSineOp(const Tensor &operand, TensorType resultType);
Tensor evalSliceOp(const Tensor &operand, Index startIndices, Sizes strides,
                   TensorType resultType);
Tensor evalSqrtOp(const Tensor &operand, TensorType resultType);
Tensor evalSubtractOp(const Tensor &lhs, const Tensor &rhs,
                      TensorType resultType);
Tensor evalTanhOp(const Tensor &operand, TensorType resultType);
Tensor evalTransposeOp(const Tensor &operand, const Axes &permutation,
                       TensorType resultType);
SmallVector<Tensor> evalWhileOp(ArrayRef<Tensor> operand, Region &cond,
                                Region &body, Scope &scope);
Tensor evalXorOp(const Tensor &lhs, const Tensor &rhs, TensorType resultType);

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
