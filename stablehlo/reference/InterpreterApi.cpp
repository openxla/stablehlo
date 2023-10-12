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
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/DebugStringHelper.h"
#include "stablehlo/reference/Errors.h"
#include "stablehlo/reference/InterpreterOps.h"
#include "stablehlo/reference/NumPy.h"
#include "stablehlo/reference/Ops.h"

namespace mlir {
namespace stablehlo {
namespace {

llvm::Error wrapStatus(llvm::Error status, llvm::StringRef funcName,
                       llvm::StringRef fallbackName) {
  if (status)
    return stablehlo::invalidArgument(
        "Error evaluating function: %s. \n\tFallback for %s failed: %s",
        funcName.data(), fallbackName.data(),
        toString(std::move(status)).c_str());
  return llvm::Error::success();
}

}  // namespace

llvm::Error InterpreterFallback::operator()(Operation &op, Process *process,
                                            Scope &scope) {
  llvm::StringRef funcName = currentFcn.getSymName();

  if (auto probeOp = dyn_cast<stablehlo::interpreter::ProbeOp>(op)) {
    auto input =
        stablehlo::InterpreterValue(scope.findTensor(probeOp.getOperand()));
    auto status = stablehlo::interpreter::evalProbeOp(
        input, probeOp.getProbeId(), config->probeInstrumentationDir,
        instrumentedTensors);
    scope.add(probeOp.getResult(), input);
    return wrapStatus(std::move(status), funcName, "interpreter.probe");
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
    return wrapStatus(llvm::Error::success(), funcName,
                      "interpreter.run_parallel");
  }

  return handleOp(op, process, scope);
}

llvm::Error InterpreterFallback::handleOp(Operation &op, Process *process,
                                          Scope &scope) {
  return stablehlo::invalidArgument("Unsupported op: %s",
                                    debugString(op).c_str());
}

llvm::ErrorOr<SmallVector<InterpreterValue>> runInterpreter(
    const std::string &mlir, ArrayRef<InterpreterValue> inputs,
    const InterpreterConfiguration &config) {
  std::unique_ptr<llvm::MemoryBuffer> program_buffer =
      llvm::MemoryBuffer::getMemBuffer(mlir);

  llvm::SourceMgr source_mgr;
  MLIRContext context;
  source_mgr.AddNewSourceBuffer(std::move(program_buffer), llvm::SMLoc());
  OwningOpRef<ModuleOp> module(parseSourceFile<ModuleOp>(source_mgr, &context));

  return runInterpreter(module.get(), inputs, config);
}

llvm::ErrorOr<SmallVector<InterpreterValue>> runInterpreter(
    ModuleOp module, ArrayRef<InterpreterValue> inputs,
    const InterpreterConfiguration &config) {
  auto numFuncs = 0;
  bool hasMain = false;

  assert(!config.mainFunction.empty() && "Main function name cannot be empty");
  module.walk([&](func::FuncOp funcOp) {
    if (funcOp.getSymName() == config.mainFunction) hasMain = true;
    numFuncs++;
  });

  if (numFuncs > 1 && !hasMain)
    llvm::report_fatal_error("Requested main function not found.");

  if (!config.probeInstrumentationDir.empty()) {
    llvm::SmallString<128> instrumentationMetadataFile(
        config.probeInstrumentationDir);
    llvm::sys::path::append(instrumentationMetadataFile,
                            stablehlo::numpy::kInstrumentationMetadataFilename);
    if (llvm::sys::fs::remove(instrumentationMetadataFile))
      llvm::report_fatal_error(
          "Failed to remove existing instrumentation metadata file.");
  }

  SmallVector<InterpreterValue> results;
  llvm::function_ref<llvm::Error(Operation &, Process *, Scope &)> fallback =
      InterpreterFallback();

  if (config.fallback) {
    config.fallback->setConfig(config);
    fallback = *config.fallback;
  }

  auto walkResult = module.walk([&](func::FuncOp funcOp) {
    if (numFuncs > 1 && funcOp.getSymName() != config.mainFunction)
      return WalkResult::advance();

    if (config.fallback) {
      config.fallback->setFcn(funcOp);
    }

    results = stablehlo::eval(funcOp.getBody(), inputs, /*process=*/nullptr,
                              /*parent=*/nullptr, fallback);

    if (config.stream) {
      for (auto &result : results) result.print(*config.stream);
    }
    return WalkResult::advance();
  });

  if (walkResult.wasInterrupted()) {
    return llvm::errc::interrupted;
  }
  return results;
}

}  // namespace stablehlo
}  // namespace mlir
