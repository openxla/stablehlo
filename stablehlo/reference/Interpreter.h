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

#ifndef STABLEHLO_REFERENCE_INTERPRETER_H
#define STABLEHLO_REFERENCE_INTERPRETER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "stablehlo/reference/Tensor.h"

namespace mlir {
namespace stablehlo {

/// Evaluating an mlir function.
///
/// Assuming that the function under evaluation has passed verifier,
/// similarly to what's required by constant folding.
llvm::Expected<SmallVector<Tensor>> eval(func::FuncOp func,
                                         ArrayRef<Tensor> args);

}  // namespace stablehlo
}  // namespace mlir

#endif  // STABLEHLO_REFERENCE_INTERPRETER_H
