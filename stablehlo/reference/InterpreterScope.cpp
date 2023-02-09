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

#include "InterpreterScope.h"

#include "llvm/Support/FormatVariadic.h"
#include "mlir/Support/DebugStringHelper.h"

namespace mlir {
namespace stablehlo {

void InterpreterScope::add(Value ssaValue, Tensor runtimeValue) {
  // We are instantiating a new `InterpreterScope` object every time the
  // interpreter evaluates a region. With that, the `stackFrame` should not have
  // any duplicates.
  assert(!stackFrame.count(ssaValue));
  stackFrame[ssaValue] = runtimeValue;
}

Tensor InterpreterScope::find(Value ssaValue) const {
  auto it = stackFrame.find(ssaValue);

  if (it != stackFrame.end()) {
    return it->second;
  }

  if (!parentScope)
    llvm::report_fatal_error(
        llvm::formatv("Expected the value {0} to be already evaluated",
                      debugString(ssaValue).c_str()));

  return parentScope->find(ssaValue);
}

}  // namespace stablehlo
}  // namespace mlir
