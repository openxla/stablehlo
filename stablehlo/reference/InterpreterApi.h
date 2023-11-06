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

#ifndef STABLEHLO_REFERENCE_INTERPRETERAPI_H
#define STABLEHLO_REFERENCE_INTERPRETERAPI_H

#include "llvm/Support/ErrorOr.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/reference/InterpreterConfiguration.h"
#include "stablehlo/reference/InterpreterValue.h"

namespace mlir {
namespace stablehlo {

/// Invoke the StableHLO reference interpreter with the given parsed MLIR
/// module input and provided inputs. Returns a list of interpreter outputs.
/// Can optionally pass a fallback interpreter callback which executes when no
/// builtin kernels are matched.
llvm::ErrorOr<SmallVector<InterpreterValue>> evalModule(
    ModuleOp module, ArrayRef<InterpreterValue> inputs,
    const InterpreterConfiguration &config);

}  // namespace stablehlo
}  // namespace mlir

#endif  // STABLEHLO_REFERENCE_INTERPRETERAPI_H
