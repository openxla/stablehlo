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

#include <functional>
#include <iostream>
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/ToolUtilities.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/reference/Interpreter.h"
#include "stablehlo/reference/Tensor.h"

using namespace llvm;
using namespace mlir;

namespace {

/// Parses the source program represented by the memory buffer, runs the
/// interpreter on the parsed function, and finally prints the evaluated value
/// to os.
static LogicalResult processBuffer(std::unique_ptr<MemoryBuffer> chunkBuffer,
                                   raw_ostream &os) {
  // Registers Dialects and sets up the MLIR context.
  DialectRegistry registry;
  registry.insert<func::FuncDialect>();
  registry.insert<stablehlo::StablehloDialect>();
  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(chunkBuffer), SMLoc());
  SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);

  // Prepare the input buffer.
  ParserConfig config(&context);
  OwningOpRef<ModuleOp> module(parseSourceFile<ModuleOp>(sourceMgr, config));
  if (!module) return failure();

  for (auto funcOp : module->getBodyRegion().getOps<mlir::func::FuncOp>()) {
    os << "\nEvaluated results of function: " << funcOp.getSymName() << "\n";

    // Run the test model.
    auto results = mlir::stablehlo::eval(funcOp, {});
    if (!(bool)results) toString(results.takeError());

    // Dump the results.
    for(auto &result : *results) result.print(os);
  }

  return success();
}

LogicalResult InterpreterMain(int argc, char **argv) {
  static cl::opt<std::string> inputFilename(
      cl::Positional, cl::desc("<input file>"), cl::init("-"));

  static cl::opt<std::string> outputFilename("o", cl::desc("Output filename"),
                                             cl::value_desc("filename"),
                                             cl::init("-"));

  static cl::opt<bool> splitInputFile(
      "split-input-file",
      cl::desc("Split the input file into pieces and process each "
               "chunk independently"),
      cl::init(false));

  InitLLVM y(argc, argv);
  cl::ParseCommandLineOptions(argc, argv, "StableHLO interpreter test runner");

  // Set up the input/output files.
  std::string errorMessage;
  auto file = openInputFile(inputFilename, &errorMessage);
  if (!file) {
    llvm::errs() << errorMessage << "\n";
    return failure();
  }

  auto output = openOutputFile(outputFilename, &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << "\n";
    return failure();
  }

  // Process the input buffer.
  if (failed(splitAndProcessBuffer(std::move(file), processBuffer, output->os(),
                                   splitInputFile,
                                   /*insertMarkerInOutput=*/true)))
    return failure();

  // Keep the output file if the invocation of InterpreterMain was successful.
  output->keep();
  return success();
}

}  // namespace

int main(int argc, char **argv) { return failed(InterpreterMain(argc, argv)); }
