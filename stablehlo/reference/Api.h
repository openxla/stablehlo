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

#ifndef STABLEHLO_REFERENCE_API_H
#define STABLEHLO_REFERENCE_API_H

#include <string>

#include "llvm/Support/ErrorOr.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/reference/Configuration.h"
#include "stablehlo/reference/Value.h"

namespace mlir {
namespace stablehlo {

/// Invoke the StableHLO reference interpreter with the given parsed MLIR
/// module input and provided inputs. Returns a list of interpreter outputs.
/// Can optionally pass a fallback interpreter callback which executes when no
/// builtin kernels are matched.
FailureOr<SmallVector<InterpreterValue>> evalModule(
    ModuleOp module, ArrayRef<InterpreterValue> inputs,
    const InterpreterConfiguration &config);

/// This wrapper is intended to be easily used by the StableHLO Python bindings.
// It wraps the InterpreterValue API.
FailureOr<SmallVector<DenseElementsAttr>> evalModule(
    ModuleOp module, ArrayRef<DenseElementsAttr> inputs,
    const InterpreterConfiguration &config);

/// Parses a StableHLO MLIR text program into a ModuleOp.
FailureOr<OwningOpRef<ModuleOp>> parseStablehloModule(const std::string &mlir,
                                                      MLIRContext &context);

}  // namespace stablehlo
}  // namespace mlir

#endif  // STABLEHLO_REFERENCE_API_H
