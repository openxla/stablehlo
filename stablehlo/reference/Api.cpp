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
#include "stablehlo/reference/Api.h"

#include <cstdint>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "stablehlo/dialect/Register.h"
#include "stablehlo/reference/Configuration.h"
#include "stablehlo/reference/Errors.h"
#include "stablehlo/reference/InterpreterOps.h"
#include "stablehlo/reference/NumPy.h"
#include "stablehlo/reference/Ops.h"
#include "stablehlo/reference/Process.h"
#include "stablehlo/reference/Scope.h"
#include "stablehlo/reference/Tensor.h"
#include "stablehlo/reference/Value.h"
#include "stablehlo/transforms/Passes.h"

namespace mlir {
namespace stablehlo {
namespace {
FailureOr<func::FuncOp> getMainFunction(ModuleOp module, StringRef mainName) {
  auto functions = module.getOps<func::FuncOp>();

  for (auto funcOp : functions)
    if (funcOp.getSymName() == mainName) return funcOp;

  bool isSingleFunction =
      std::distance(functions.begin(), functions.end()) == 1;
  bool isDefaultLookup = mainName == "main";
  if (isSingleFunction && isDefaultLookup) return *functions.begin();

  return module.emitError()
         << "module must have entry func with name " << mainName;
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
          ++serializedProbeFileId);
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
          infeed.push(cast<FlatSymbolRefAttr>(value).getAttr());

      SmallVector<SmallVector<StringAttr>> programs(
          runParallelOp.getPrograms().size());
      for (auto [i, replica] : llvm::enumerate(runParallelOp.getPrograms()))
        for (auto &program : cast<ArrayAttr>(replica))
          programs[i].push_back(cast<FlatSymbolRefAttr>(program).getAttr());

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

  /// Probe instrumentation counter for uniquely identifying instrumented tensor
  /// filenames.
  int64_t serializedProbeFileId = 0;
};

LogicalResult validateEntrySignature(func::FuncOp func,
                                     ArrayRef<InterpreterValue> inputs) {
  if (func.getNumArguments() != inputs.size())
    return func->emitError()
           << "incorrect number of arguments specified, provided "
           << inputs.size() << " inputs but function expected"
           << func.getNumArguments();

  TypeRange signature = func.getArgumentTypes();
  for (auto [i, sigType, arg] : llvm::enumerate(signature, inputs)) {
    auto argType = arg.getType();
    if (sigType != argType)
      return func.emitError() << "invalid input argument type at index " << i
                              << ", input type was " << argType
                              << " but entry function expected " << sigType;
  }
  return success();
}

// Specializes the shapes of arguments in function 'func' based on runtime input
// values. If all argument types already have static shapes, this function does
// nothing. Otherwise, it constructs a pipeline of MLIR passes to refine
// argument shapes using the provided `inputs`.
//
// Args:
//   module: The MLIR module containing the function.
//   func: The function whose argument shapes need specialization.
//   inputs: The runtime input values.
//
// Returns:
//   A `LogicalResult` indicating success or failure of the shape
//   refinement pipeline.
LogicalResult removeDynamism(ModuleOp module, func::FuncOp func,
                             ArrayRef<InterpreterValue> inputs) {
  if (llvm::all_of(func.getArgumentTypes(), [](Type type) {
        return llvm::cast<ShapedType>(type).hasStaticShape();
      })) {
    return success();
  }
  SmallVector<Type> refinedTypes = llvm::to_vector(llvm::map_range(
      inputs, [](InterpreterValue input) { return input.getType(); }));

  PassManager pm(module.getContext());
  stablehlo::createStablehloRemoveDynamismPipeline(pm, refinedTypes);
  if (failed(pm.run(module))) {
    return func.emitError("Failed to refine dynamic shape in function: ")
           << func.getName();
  }
  return success();
}

}  // namespace

FailureOr<SmallVector<InterpreterValue>> evalModule(
    ModuleOp module, ArrayRef<InterpreterValue> inputs,
    const InterpreterConfiguration &config) {
  // Additional error checking at main function boundary.
  // This is most likely user error, where future errors during interpreting are
  // more likely invalid IR or interpreter bugs.
  if (module.getOps<func::FuncOp>().empty())
    return SmallVector<InterpreterValue>();

  auto mainFunc = getMainFunction(module, config.mainFunction);
  if (failed(mainFunc) || failed(removeDynamism(module, *mainFunc, inputs)) ||
      failed(validateEntrySignature(*mainFunc, inputs))) {
    return failure();
  }

  if (!config.probeInstrumentationDir.empty()) {
    llvm::SmallString<128> instrumentationMetadataFile(
        config.probeInstrumentationDir);
    llvm::sys::path::append(instrumentationMetadataFile,
                            stablehlo::numpy::kInstrumentationMetadataFilename);
    if (llvm::sys::fs::remove(instrumentationMetadataFile))
      return emitError(
          UnknownLoc::get(module.getContext()),
          "Failed to remove existing instrumentation metadata file.");
  }

  DefaultInterpreterFallback fallback(config);
  return stablehlo::eval(mainFunc->getBody(), inputs, &fallback);
}

FailureOr<SmallVector<DenseElementsAttr>> evalModule(
    ModuleOp module, ArrayRef<DenseElementsAttr> inputs,
    const InterpreterConfiguration &config) {
  SmallVector<InterpreterValue> valueInputs = llvm::map_to_vector(
      inputs, [](DenseElementsAttr attr) -> InterpreterValue {
        return InterpreterValue(makeTensor(attr));
      });

  auto values = evalModule(module, valueInputs, config);
  if (failed(values)) return failure();

  SmallVector<DenseElementsAttr> results = llvm::map_to_vector(
      values.value(), [](InterpreterValue val) -> DenseElementsAttr {
        return makeDenseElementsAttr(val.getTensor());
      });

  return results;
}

FailureOr<OwningOpRef<ModuleOp>> parseStablehloModule(const std::string &mlir,
                                                      MLIRContext &context) {
  llvm::SourceMgr source_mgr;
  source_mgr.AddNewSourceBuffer(llvm::MemoryBuffer::getMemBuffer(mlir),
                                llvm::SMLoc());

  mlir::DialectRegistry registry;
  mlir::stablehlo::registerAllDialects(registry);
  context.loadDialect<mlir::stablehlo::interpreter::InterpreterDialect>();
  context.appendDialectRegistry(registry);

  mlir::OwningOpRef<mlir::ModuleOp> module(
      mlir::parseSourceFile<mlir::ModuleOp>(source_mgr, &context));

  if (!module)
    return emitError(UnknownLoc::get(&context), "unable to parse module");

  return module;
}

}  // namespace stablehlo
}  // namespace mlir
