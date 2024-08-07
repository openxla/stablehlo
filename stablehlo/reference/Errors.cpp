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

#include "stablehlo/reference/Errors.h"

#include <utility>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

namespace mlir {
namespace stablehlo {

llvm::Error wrapFallbackStatus(llvm::Error status, llvm::StringRef funcName,
                               llvm::StringRef fallbackName) {
  if (status)
    return stablehlo::invalidArgument(
        "Error evaluating function: %s. \n\tFallback for %s failed: %s",
        funcName.data(), fallbackName.data(),
        toString(std::move(status)).c_str());
  return llvm::Error::success();
}

}  // namespace stablehlo
}  // namespace mlir
