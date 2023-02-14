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

#include "Scope.h"

#include "llvm/Support/FormatVariadic.h"
#include "mlir/Support/DebugStringHelper.h"

namespace mlir {
namespace stablehlo {

void Scope::add(Value ssaValue, Tensor runtimeValue) {
  // We are instantiating a new `Scope` object every time the
  // interpreter evaluates a region. With that, the `stack_frame_` should not
  // have any duplicates.
  if (stack_frame_.count(ssaValue))
    llvm::report_fatal_error("Duplicate SSA register found in scope");

  if (ssaValue.getType() != runtimeValue.getType())
    llvm::report_fatal_error(
        "Expected same type for an SSA register and its evaluated value");

  stack_frame_[ssaValue] = runtimeValue;
}

void Scope::add(ValueRange ssaValues, ArrayRef<Tensor> runtimeValues) {
  assert(ssaValues.size() == runtimeValues.size());
  for (auto [ssaValue, runtimeValue] : llvm::zip(ssaValues, runtimeValues))
    add(ssaValue, runtimeValue);
}

Tensor Scope::find(Value ssaValue) const {
  auto it = stack_frame_.find(ssaValue);

  if (it != stack_frame_.end()) return it->second;

  if (!parent_)
    llvm::report_fatal_error(llvm::formatv("value {0} not found in scope",
                                           debugString(ssaValue).c_str()));

  return parent_->find(ssaValue);
}

SmallVector<Tensor> Scope::find(ValueRange ssaValues) const {
  return llvm::to_vector(
      llvm::map_range(ssaValues, [&](Value value) { return find(value); }));
}

}  // namespace stablehlo
}  // namespace mlir
