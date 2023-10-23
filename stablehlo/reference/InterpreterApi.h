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

#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorOr.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/reference/InterpreterValue.h"
#include "stablehlo/reference/Process.h"
#include "stablehlo/reference/Scope.h"

namespace mlir {
namespace stablehlo {

class InterpreterFallback;
struct InterpreterConfiguration {
 public:
  InterpreterConfiguration()
      : fallback(std::make_unique<InterpreterFallback>()) {}

  /// If specified, the directory to which StableHLO interpreter tensors will
  /// be serialized to disk.
  std::string probeInstrumentationDir = "";

  /// If specified, use this function as the entrypoint function into the model.
  /// Will otherwise default to “main”.
  std::string mainFunction = "main";

  /// If specified, use the callback to run on ops which do not have a
  /// registered kernel.
  std::unique_ptr<InterpreterFallback> fallback;

  /// If set, optionally dump tensor values to the specified stream.
  raw_ostream *stream = nullptr;
};

/// Base interpreter fallback callback functor to run when no registered kernels
/// are found for a given StableHLO operation.
class InterpreterFallback {
 public:
  llvm::Error operator()(Operation &op, Process *process, Scope &scope);

  virtual ~InterpreterFallback() = default;

  /// Set the user provided interpreter configuration.
  void setConfig(const InterpreterConfiguration &config) {
    this->config = &config;
  }

  /// Set the topmost currently executing function in the module.
  void setFunction(func::FuncOp function) { currentFunction = function; }

 protected:
  /// Custom op kernels for any user specified ops not found in the StableHLO
  /// op dialect or StableHLO interpreter dialect.
  virtual llvm::Error handleOp(Operation &op, Process *process, Scope &scope);

  /// The user provided interpreter configuration.
  const InterpreterConfiguration *config;

  /// The topmost function currently being executed in the module.
  func::FuncOp currentFunction;

 private:
  /// If the input StableHLO program has been instrumented, keep track of how
  /// many times a given operation has been executed.
  llvm::StringMap<int32_t> instrumentedTensors;
};

/// Invoke the StableHLO reference interpreter with the given parsed MLIR
/// module input and provided inputs. Returns a list of interpreter outputs.
/// Can optionally pass a fallback interpreter callback which executes when no
/// builtin kernels are matched.
llvm::ErrorOr<SmallVector<InterpreterValue>> runInterpreter(
    ModuleOp module, ArrayRef<InterpreterValue> inputs,
    const InterpreterConfiguration &config);

}  // namespace stablehlo
}  // namespace mlir

#endif  // STABLEHLO_REFERENCE_INTERPRETERAPI_H
