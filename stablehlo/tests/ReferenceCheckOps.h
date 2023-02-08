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

#ifndef STABLEHLO_TESTS_INTERPRETCHECKOPS_H
#define STABLEHLO_TESTS_INTERPRETCHECKOPS_H

#include "mlir/IR/BuiltinAttributes.h"
#include "stablehlo/reference/Tensor.h"
#include "stablehlo/tests/CheckOps.h"

namespace mlir {
namespace stablehlo {
namespace check {

// The eval functions for the following ops are used only for test harness.
llvm::Error evalEqOp(const Tensor &lhs, ElementsAttr value);
llvm::Error evalAlmostEqOp(const Tensor &lhs, ElementsAttr value);

}  // namespace check
}  // namespace stablehlo
}  // namespace mlir

#endif  // STABLEHLO_TESTS_INTERPRETCHECKOPS_H
