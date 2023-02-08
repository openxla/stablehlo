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

#include "stablehlo/tests/ReferenceCheckOps.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/DebugStringHelper.h"
#include "stablehlo/reference/Errors.h"
#include "stablehlo/reference/Tensor.h"

namespace mlir {
namespace stablehlo {
namespace check {

llvm::Error evalEqOp(const Tensor &lhs, ElementsAttr value) {
  auto rhs = makeTensor(value.cast<DenseElementsAttr>());
  for (auto lhsIt = lhs.index_begin(), rhsIt = rhs.index_begin();
       lhsIt != lhs.index_end(); ++lhsIt, ++rhsIt) {
    if (lhs.get(*lhsIt) != rhs.get(*rhsIt)) {
      return invalidArgument(
          "Element value don't match: %s (actual) vs %s (expected)\n",
          debugString(lhs.get(*lhsIt)).c_str(),
          debugString(rhs.get(*rhsIt)).c_str());
    }
  }

  return llvm::Error::success();
}

llvm::Error evalAlmostEqOp(const Tensor &lhs, ElementsAttr value) {
  auto rhs = makeTensor(value.cast<DenseElementsAttr>());
  for (auto lhsIt = lhs.index_begin(), rhsIt = rhs.index_begin();
       lhsIt != lhs.index_end(); ++lhsIt, ++rhsIt) {
    if (!areApproximatelyEqual(lhs.get(*lhsIt), rhs.get(*rhsIt))) {
      return invalidArgument(
          "Element value don't match: %s (actual) vs %s (expected)\n",
          debugString(lhs.get(*lhsIt)).c_str(),
          debugString(rhs.get(*rhsIt)).c_str());
    }
  }

  return llvm::Error::success();
}

}  // namespace check
}  // namespace stablehlo
}  // namespace mlir
