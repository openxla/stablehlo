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
#include "stablehlo/reference/InterpreterApi.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "stablehlo/reference/Errors.h"
#include "stablehlo/reference/InterpreterConfiguration.h"
#include "stablehlo/reference/InterpreterOps.h"
#include "stablehlo/reference/NumPy.h"
#include "stablehlo/reference/Ops.h"
#include "stablehlo/reference/Process.h"
#include "stablehlo/reference/Scope.h"

namespace mlir {
namespace stablehlo {
namespace {
func::FuncOp getMainFunction(ModuleOp module, StringRef mainName) {
  auto functions = module.getOps<func::FuncOp>();

  for (auto funcOp : functions)
    if (funcOp.getSymName().equals(mainName)) return funcOp;

  bool isSingleFunction =
      std::distance(functions.begin(), functions.end()) == 1;
  bool isDefaultLookup = mainName == "main";
  if (isSingleFunction && isDefaultLookup) return *functions.begin();

  return {};
}

// DefaultInterpreterFallback is an implementation detail of run module. It
// takes in an InterpreterConfiguration which can have user-implemented
// fallbacks.
class DefaultInterpreterFallback : public InterpreterFallback {
 public:
  DefaultInterpreterFallback(const InterpreterConfiguration &config)
      : config(config){};

  virtual llvm::Error operator()(Operation &op, Scope &scope,
                                 Process *process) final {
    llvm::StringRef funcName = op.getParentOfType<func::FuncOp>().getSymName();

    if (auto probeOp = dyn_cast<stablehlo::interpreter::ProbeOp>(op)) {
      auto input =
          stablehlo::InterpreterValue(scope.findTensor(probeOp.getOperand()));
      auto status = stablehlo::interpreter::evalProbeOp(
          input, probeOp.getProbeId(), config.probeInstrumentationDir,
          instrumentedTensors);
      scope.add(probeOp.getResult(), input);
      return wrapFallbackStatus(std::move(status), funcName,
                                "interpreter.probe");
    }

    if (auto runParallelOp =
            dyn_cast<stablehlo::interpreter::RunParallelOp>(op)) {
      auto runtimeOperands = scope.find(runParallelOp.getInputs());
      std::queue<StringAttr> infeed;
      if (auto infeedAttr = runParallelOp.getInfeed())
        for (auto &value : infeedAttr->getValue())
          infeed.push(value.cast<FlatSymbolRefAttr>().getAttr());

      SmallVector<SmallVector<StringAttr>> programs(
          runParallelOp.getPrograms().size());
      for (auto [i, replica] : llvm::enumerate(runParallelOp.getPrograms()))
        for (auto &program : replica.cast<ArrayAttr>())
          programs[i].push_back(program.cast<FlatSymbolRefAttr>().getAttr());

      SymbolTable symbolTable{op.getParentOfType<ModuleOp>()};
      auto results = stablehlo::interpreter::evalRunParallelOp(
          runtimeOperands, infeed, programs, symbolTable);
      scope.add(runParallelOp.getResults(), results);
      return wrapFallbackStatus(llvm::Error::success(), funcName,
                                "interpreter.run_parallel");
    }

    return (*config.fallback)(op, scope, process);
  }

 private:
  /// Interpreter configuration.
  const InterpreterConfiguration &config;

  /// If the input StableHLO program has been instrumented, keep track of how
  /// many times a given operation has been executed.
  llvm::StringMap<int32_t> instrumentedTensors;
};

}  // namespace

llvm::ErrorOr<SmallVector<InterpreterValue>> evalModule(
    ModuleOp module, ArrayRef<InterpreterValue> inputs,
    const InterpreterConfiguration &config) {
  auto mainFunc = getMainFunction(module, config.mainFunction);
  if (!mainFunc) llvm::report_fatal_error("Requested main function not found.");

  if (!config.probeInstrumentationDir.empty()) {
    llvm::SmallString<128> instrumentationMetadataFile(
        config.probeInstrumentationDir);
    llvm::sys::path::append(instrumentationMetadataFile,
                            stablehlo::numpy::kInstrumentationMetadataFilename);
    if (llvm::sys::fs::remove(instrumentationMetadataFile))
      llvm::report_fatal_error(
          "Failed to remove existing instrumentation metadata file.");
  }

  DefaultInterpreterFallback fallback(config);
  return stablehlo::eval(mainFunc.getBody(), inputs, &fallback);
}

}  // namespace stablehlo
}  // namespace mlir
