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

#include "stablehlo/reference/Configuration.h"

#include "llvm/Support/Error.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/DebugStringHelper.h"
#include "stablehlo/reference/Errors.h"
#include "stablehlo/reference/Process.h"
#include "stablehlo/reference/Scope.h"

namespace mlir {
namespace stablehlo {

llvm::Error InterpreterFallback::operator()(Operation &op, Scope &scope,
                                            Process *process) {
  return stablehlo::invalidArgument("Unsupported op: %s",
                                    debugString(op).c_str());
}

}  // namespace stablehlo
}  // namespace mlir
