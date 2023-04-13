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
Tensor evalBroadcastInDimOp(const Tensor &operand, Axes broadcastDimensions,
                            ShapedType resultType);
SmallVector<Tensor> evalCaseOp(const Tensor &index, RegionRange branches,
                               Scope &scope);
Tensor evalCeilOp(const Tensor &operand, ShapedType resultType);
Tensor evalClampOp(const Tensor &min, const Tensor &operand, const Tensor &max,
                   ShapedType resultType);
Tensor evalClzOp(const Tensor &operand, ShapedType resultType);
Tensor evalCompareOp(const Tensor &lhs, const Tensor &rhs,
                     ComparisonDirection comparisonDirection,
                     ShapedType resultType);
Tensor evalConcatenateOp(ArrayRef<Tensor> inputs, Axis dimension,
                         ShapedType resultType);
Tensor evalConstantOp(ElementsAttr value);
Tensor evalConvertOp(const Tensor &operand, ShapedType resultType);
Tensor evalCosineOp(const Tensor &operand, ShapedType resultType);
Tensor evalDivideOp(const Tensor &lhs, const Tensor &rhs,
                    ShapedType resultType);
Tensor evalDynamicSliceOp(const Tensor &operand, ArrayRef<Tensor> startIndices,
                          Sizes sliceSizes, ShapedType resultType);
Tensor evalDynamicUpdateSliceOp(const Tensor &operand, const Tensor &update,
                                ArrayRef<Tensor> startIndices,
                                ShapedType resultType);
Tensor evalExponentialOp(const Tensor &operand, ShapedType resultType);
Tensor evalFloorOp(const Tensor &operand, ShapedType resultType);
SmallVector<Tensor> evalIfOp(const Tensor &pred, Region &trueBranch,
                             Region &falseBranch, Scope &scope);
Tensor evalImagOp(const Tensor &operand, ShapedType resultType);
Tensor evalIotaOp(Axis iotaDimension, ShapedType resultType);
Tensor evalIsFiniteOp(const Tensor &operand, ShapedType resultType);
Tensor evalLogOp(const Tensor &operand, ShapedType resultType);
Tensor evalLogisticOp(const Tensor &operand, ShapedType resultType);
Tensor evalMaxOp(const Tensor &lhs, const Tensor &rhs, ShapedType resultType);
Tensor evalMinOp(const Tensor &lhs, const Tensor &rhs, ShapedType resultType);
Tensor evalMultiplyOp(const Tensor &lhs, const Tensor &rhs,
                      ShapedType resultType);
Tensor evalNegOp(const Tensor &operand, ShapedType resultType);
Tensor evalNotOp(const Tensor &operand, ShapedType resultType);
Tensor evalOrOp(const Tensor &lhs, const Tensor &rhs, ShapedType resultType);
Tensor evalPadOp(const Tensor &operand, const Tensor &paddingValue,
                 Sizes edgePaddingLow, Sizes interiorPadding,
                 ShapedType resultType);
Tensor evalPowerOp(const Tensor &lhs, const Tensor &rhs, ShapedType resultType);
Tensor evalRealOp(const Tensor &operand, ShapedType resultType);
Tensor evalRemOp(const Tensor &lhs, const Tensor &rhs, ShapedType resultType);
Tensor evalReshapeOp(const Tensor &operand, ShapedType resultType);
Tensor evalReverseOp(const Tensor &operand, Axes dimensions,
                     ShapedType resultType);
Tensor evalRsqrtOp(const Tensor &operand, ShapedType resultType);
Tensor evalSelectOp(const Tensor &pred, const Tensor &onTrue,
                    const Tensor &onFalse, ShapedType resultType);
Tensor evalSineOp(const Tensor &operand, ShapedType resultType);
Tensor evalSliceOp(const Tensor &operand, Index startIndices, Sizes strides,
                   ShapedType resultType);
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
