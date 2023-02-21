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
Tensor evalAbsOp(const Tensor &operand, Type resultType);
Tensor evalAddOp(const Tensor &lhs, const Tensor &rhs, Type resultType);
Tensor evalAndOp(const Tensor &lhs, const Tensor &rhs, Type resultType);
Tensor evalBroadcastInDimOp(const Tensor &operand, Axes broadcastDimensions,
                            Type resultType);
Tensor evalCeilOp(const Tensor &operand, Type resultType);
Tensor evalClampOp(const Tensor &min, const Tensor &operand, const Tensor &max,
                   Type resultType);
Tensor evalConstantOp(ElementsAttr value);
Tensor evalConvertOp(const Tensor &operand, Type resultType);
Tensor evalCosineOp(const Tensor &operand, Type resultType);
Tensor evalDynamicSliceOp(const Tensor &operand, ArrayRef<Tensor> startIndices,
                          Sizes sliceSizes, Type resultType);
Tensor evalDynamicUpdateSliceOp(const Tensor &operand, const Tensor &update,
                                ArrayRef<Tensor> startIndices, Type resultType);
Tensor evalExponentialOp(const Tensor &operand, Type resultType);
Tensor evalFloorOp(const Tensor &operand, Type resultType);
SmallVector<Tensor> evalIfOp(const Tensor &pred, Region &trueBranch,
                             Region &falseBranch, Scope &scope);
Tensor evalIotaOp(Axis iotaDimension, Type resultType);
Tensor evalLogOp(const Tensor &operand, Type resultType);
Tensor evalMaxOp(const Tensor &lhs, const Tensor &rhs, Type resultType);
Tensor evalMinOp(const Tensor &lhs, const Tensor &rhs, Type resultType);
Tensor evalMultiplyOp(const Tensor &lhs, const Tensor &rhs, Type resultType);
Tensor evalNegOp(const Tensor &operand, Type resultType);
Tensor evalNotOp(const Tensor &operand, Type resultType);
Tensor evalOrOp(const Tensor &lhs, const Tensor &rhs, Type resultType);
Tensor evalPadOp(const Tensor &operand, const Tensor &paddingValue,
                 Sizes edgePaddingLow, Sizes interiorPadding, Type resultType);
Tensor evalReshapeOp(const Tensor &operand, Type resultType);
Tensor evalReverseOp(const Tensor &operand, Axes dimensions, Type resultType);
Tensor evalSelectOp(const Tensor &pred, const Tensor &onTrue,
                    const Tensor &onFalse, Type resultType);
Tensor evalSineOp(const Tensor &operand, Type resultType);
Tensor evalSliceOp(const Tensor &operand, Sizes startIndices,
                   Sizes strides, Type resultType);
Tensor evalSqrtOp(const Tensor &operand, Type resultType);
Tensor evalSubtractOp(const Tensor &lhs, const Tensor &rhs, Type resultType);
Tensor evalTanhOp(const Tensor &operand, Type resultType);
Tensor evalTransposeOp(const Tensor &operand, const Axes &permutation,
                       Type resultType);
SmallVector<Tensor> evalWhileOp(ArrayRef<Tensor> operand, Region &cond,
                                Region &body, Scope &scope);
Tensor evalXorOp(const Tensor &lhs, const Tensor &rhs, Type resultType);

/// Evaluates an mlir::Region `region` using the runtime values `args`
/// corresponding to the arguments of the entry block of the region.
/// Interprets the operations within the entry block and returns the runtime
/// values for the terminator's arguments.
/// Assumes that the region has only one block.
llvm::SmallVector<Tensor> eval(Region &region, llvm::ArrayRef<Tensor> args,
                               Scope *parentScope = nullptr);

}  // namespace stablehlo
}  // namespace mlir

#endif  // STABLEHLO_REFERENCE_OPS_H
