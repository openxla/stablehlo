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

#include "stablehlo/reference/InterpreterOps.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <future>
#include <queue>
#include <string>

#include "llvm/Support/Casting.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "stablehlo/reference/NumPy.h"
#include "stablehlo/reference/Ops.h"
#include "stablehlo/reference/Process.h"
#include "stablehlo/reference/ProcessGrid.h"
#include "stablehlo/reference/Value.h"

#define GET_OP_CLASSES
#include "stablehlo/reference/InterpreterOps.cpp.inc"

namespace mlir {
namespace stablehlo {
namespace interpreter {
namespace {

// Appends a new line item to an instrumentation metadata file, `index.json` in
// the form: `probeId,probeOutputDir/filename`.
llvm::Error writeProbeMetadata(StringRef probeId, Type type, StringRef filename,
                               StringRef probeOutputDir) {
  if (probeOutputDir.empty())
    return createStringError(llvm::errc::invalid_argument,
                             "Probe serialization directory cannot be empty.");

  llvm::SmallString<128> filepath(probeOutputDir);
  llvm::sys::path::append(filepath, numpy::kInstrumentationMetadataFilename);

  int fd;
  if (llvm::sys::fs::openFileForWrite(filepath, fd,
                                      llvm::sys::fs::CD_CreateAlways,
                                      llvm::sys::fs::OF_Append))
    return createStringError(llvm::errc::io_error,
                             "Failed to open instrumentation metadata file.");

  llvm::raw_fd_ostream out(fd, /*shouldClose=*/true);
  out << probeId.str() << ',' << debugString(type) << ',' << filename.str()
      << '\n';

  return llvm::Error::success();
}
}  // namespace

//===----------------------------------------------------------------------===//
// Interpreter Dialect Constructor
//===----------------------------------------------------------------------===//

InterpreterDialect::InterpreterDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context,
              TypeID::get<InterpreterDialect>()) {
  addOperations<
#define GET_OP_LIST
#include "stablehlo/reference/InterpreterOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Interpreter Ops Verifier
//===----------------------------------------------------------------------===//

LogicalResult RunParallelOp::verify() {
  if (getPrograms().empty() || cast<ArrayAttr>(getPrograms()[0]).empty())
    return emitOptionalError(getLoc(), "`programs` attribute cannot be empty");

  size_t numArgs = 0;
  size_t numResults = 0;
  auto numPartitions = cast<ArrayAttr>(getPrograms()[0]).size();
  for (auto &replica : getPrograms()) {
    if (cast<ArrayAttr>(replica).size() != numPartitions)
      return emitOptionalError(
          getLoc(), "Sizes of second dimension of `programs` should all match ",
          numPartitions, " but got ", cast<ArrayAttr>(replica).size());

    for (auto &program : cast<ArrayAttr>(replica)) {
      auto funcName = cast<FlatSymbolRefAttr>(program).getAttr();
      auto func = getOperation()
                      ->getParentOfType<ModuleOp>()
                      .lookupSymbol<func::FuncOp>(funcName);
      if (!func)
        return emitOptionalError(getLoc(), "Function ", funcName, " not found");

      numArgs += func.getNumArguments();
      numResults += func.getNumResults();
    }
  }

  if (getInputs().size() != numArgs)
    return emitOptionalError(
        getLoc(), "Number of inputs (", getInputs().size(),
        ") should match the sum of the number of inputs of all programs (",
        numArgs, ")");

  if (getResults().size() != numResults)
    return emitOptionalError(
        getLoc(), "Number of results (", getResults().size(),
        ") should match the sum of the number of results of all programs (",
        numResults, ")");

  if (const auto &infeed = getInfeed()) {
    if (infeed->empty())
      return emitOptionalError(
          getLoc(), "infeed attribute is optional or should not be empty");

    for (auto flatSymbolRefAttr : infeed->getAsRange<FlatSymbolRefAttr>()) {
      auto funcName = flatSymbolRefAttr.getAttr();
      auto func = getOperation()
                      ->getParentOfType<ModuleOp>()
                      .lookupSymbol<func::FuncOp>(funcName);
      if (!func)
        return emitOptionalError(getLoc(), "Function ", funcName, " not found");

      if (func.getNumResults() != 1)
        return emitOptionalError(getLoc(), "Function ", funcName,
                                 " should return 1 tensor but returns ",
                                 func.getNumResults());

      auto resultType = func.getResultTypes()[0];
      if (!isa<ShapedType>(resultType))
        return emitOptionalError(
            getLoc(), "Function ", funcName,
            " should return a tensor type, but instead returns ", resultType);
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Interpreter Ops Evaluator
//===----------------------------------------------------------------------===//

SmallVector<InterpreterValue> evalRunParallelOp(
    ArrayRef<InterpreterValue> inputs, std::queue<StringAttr> &infeed,
    SmallVector<SmallVector<StringAttr>> programs, SymbolTable &symbolTable) {
  llvm::DefaultThreadPool threadPool;
  SmallVector<std::shared_future<SmallVector<InterpreterValue>>> futures;

  uint32_t numReplicas = programs.size();
  uint32_t numPartitions = programs[0].size();
  ProcessGrid processGrid(numReplicas, numPartitions, infeed);

  auto inputsIt = inputs.begin();

  for (uint32_t i = 0; i < numReplicas; ++i) {
    for (uint32_t j = 0; j < numPartitions; ++j) {
      auto funcName = programs[i][j];
      auto func = llvm::cast<func::FuncOp>(symbolTable.lookup(funcName));
      auto evalWrapper = [&](Region &region, ArrayRef<InterpreterValue> args,
                             ProcessId processId) {
        Process process{processId, &processGrid};
        return eval(region, args, /*config=*/nullptr, &process,
                    /*parent=*/nullptr);
      };

      auto numArgs = func.getBody().front().getArguments().size();
      SmallVector<InterpreterValue> args(inputsIt, inputsIt + numArgs);
      inputsIt += numArgs;

      futures.emplace_back(threadPool.async(
          evalWrapper, std::ref(func.getBody()), args, ProcessId{i, j}));
    }
  }

  SmallVector<InterpreterValue> results;
  for (auto &future : futures) results.append(future.get());
  // TODO(#1725): Figure out how to test the outfeed queue.
  return results;
}

// `serializedProbeFileId` should be a unique positive integer which can be used
// to unambiguously derive a serialized filename for a given `probeId`.
llvm::Error evalProbeOp(InterpreterValue input, StringRef probeId,
                        StringRef probeOutputDir,
                        int64_t serializedProbeFileId) {
  llvm::SmallString<128> filepath(probeOutputDir);

  // Use an increasing unique integer to write to disk to avoid any odd file
  // names as a result of unsafe probe_id values.
  llvm::sys::path::append(
      filepath, "probe" + std::to_string(serializedProbeFileId) + ".npy");
  auto tensor = input.getTensor();
  if (auto serializationResultError =
          numpy::serializeTensor(filepath, tensor.getType(), tensor.getData()))
    return serializationResultError;

  // After the tensor has been serialized to disk, append it to a metadata file
  // to associate the serialized probe_id with the filepath. By default, this
  // will live in an `index.csv` file generated in specified `probeOutputDir`.
  return writeProbeMetadata(probeId, input.getType(), filepath, probeOutputDir);
}

}  // namespace interpreter
}  // namespace stablehlo
}  // namespace mlir
