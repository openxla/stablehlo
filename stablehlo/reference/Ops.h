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
#include "stablehlo/reference/Scope.h"
#include "stablehlo/reference/Tensor.h"

namespace mlir {
namespace stablehlo {

// Evaluators for StableHLO ops.
Tensor evalAddOp(const Tensor &lhs, const Tensor &rhs, Type resultType);
Tensor evalAndOp(const Tensor &lhs, const Tensor &rhs, Type resultType);
Tensor evalBroadcastInDimOp(const Tensor &operand,
                            ArrayRef<int64_t> broadcastDimensions,
                            Type resultType);
Tensor evalCeilOp(const Tensor &operand, Type resultType);
Tensor evalConstantOp(ElementsAttr value);
Tensor evalConvertOp(const Tensor &operand, Type resultType);
Tensor evalCosineOp(const Tensor &operand, Type resultType);
Tensor evalDynamicSliceOp(const Tensor &operand, ArrayRef<Tensor> startIndices,
                          ArrayRef<int64_t> sliceSizes, Type resultType);
Tensor evalDynamicUpdateSliceOp(const Tensor &operand, const Tensor &update,
                                ArrayRef<Tensor> startIndices, Type resultType);
Tensor evalFloorOp(const Tensor &operand, Type resultType);
SmallVector<Tensor> evalIfOp(const Tensor &pred, Region &trueBranch,
                             Region &falseBranch, Scope &scope);
Tensor evalIotaOp(int64_t iotaDimension, Type resultType);
Tensor evalMaxOp(const Tensor &lhs, const Tensor &rhs, Type resultType);
Tensor evalMinOp(const Tensor &lhs, const Tensor &rhs, Type resultType);
Tensor evalMultiplyOp(const Tensor &lhs, const Tensor &rhs, Type resultType);
Tensor evalNegOp(const Tensor &operand, Type resultType);
Tensor evalNotOp(const Tensor &operand, Type resultType);
Tensor evalOrOp(const Tensor &lhs, const Tensor &rhs, Type resultType);
Tensor evalPadOp(const Tensor &operand, const Tensor &paddingValue,
                 ArrayRef<int64_t> edgePaddingLow,
                 ArrayRef<int64_t> interiorPadding, Type resultType);
Tensor evalReshapeOp(const Tensor &operand, Type resultType);
Tensor evalReverseOp(const Tensor &operand, ArrayRef<int64_t> dimensions,
                     Type resultType);
Tensor evalSineOp(const Tensor &operand, Type resultType);
Tensor evalSliceOp(const Tensor &operand, ArrayRef<int64_t> startIndices,
                   ArrayRef<int64_t> strides, Type resultType);
Tensor evalSubtractOp(const Tensor &lhs, const Tensor &rhs, Type resultType);
Tensor evalTanhOp(const Tensor &operand, Type resultType);
Tensor evalTransposeOp(const Tensor &operand, ArrayRef<int64_t> permutation,
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
