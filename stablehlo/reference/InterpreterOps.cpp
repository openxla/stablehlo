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

#include <future>
#include <map>
#include <thread>

#include "llvm/Support/ThreadPool.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/reference/InterpreterValue.h"
#include "stablehlo/reference/Ops.h"
#include "stablehlo/reference/Process.h"

#define GET_OP_CLASSES
#include "stablehlo/reference/InterpreterOps.cpp.inc"

namespace mlir {
namespace stablehlo {
namespace interpreter {

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

LogicalResult verifyRunParallelOp(RunParallelOp op,
                                  std::optional<Location> location,
                                  ValueRange inputs, ArrayAttr programs,
                                  int64_t numReplicas, int64_t numPartitions) {
  size_t argsSize = 0;
  for (auto &program : programs) {
    auto funcName = program.cast<StringAttr>().strref();
    auto func =
        op->getParentOfType<ModuleOp>().lookupSymbol<func::FuncOp>(funcName);
    if (!func)
      return emitOptionalError(location, "Function \"", funcName,
                               "\" not found");

    argsSize += func.getNumArguments();
  }

  if (inputs.size() != argsSize)
    return emitOptionalError(
        location, "The inputs size: ", inputs.size(),
        " does not match sum of all inputs of programs: ", argsSize);

  if (programs.size() != (size_t)numReplicas * numPartitions)
    return emitOptionalError(location,
                             "Number of programs should match numReplicas * "
                             "numPartitions (",
                             numReplicas, " * ", numPartitions, ") but got ",
                             programs.size());

  return success();
}

LogicalResult RunParallelOp::verify() {
  return verifyRunParallelOp(*this, getLoc(), getInputs(), getPrograms(),
                             getNumReplicas(), getNumPartitions());
}

//===----------------------------------------------------------------------===//
// Interpreter Ops Evaluator
//===----------------------------------------------------------------------===//

SmallVector<InterpreterValue> evalRunParallelOp(
    ArrayRef<InterpreterValue> inputs, ArrayRef<StringRef> programs,
    uint32_t numReplicas, uint32_t numPartitions, Operation &op) {
  llvm::ThreadPool threadPool;
  SmallVector<std::shared_future<SmallVector<InterpreterValue>>> futures;
  SmallVector<std::thread> processMap;
  for (uint32_t i = 0; i < numReplicas; ++i) {
    for (uint32_t j = 0; j < numPartitions; ++j) {
      auto funcName = programs[i * numPartitions + j];
      auto func =
          op.getParentOfType<ModuleOp>().lookupSymbol<func::FuncOp>(funcName);

      auto evalWrapper = [](Region &region, ArrayRef<InterpreterValue> args,
                            ProcessId processId) {
        Process process{processId};
        return eval(region, args, &process, /*parent=*/nullptr,
                    /*fallback=*/nullptr);
      };

      futures.emplace_back(threadPool.async(
          evalWrapper, std::ref(func.getBody()), inputs, ProcessId{i, j}));
    }
  }

  SmallVector<InterpreterValue> results;
  for (auto &future : futures) results.append(future.get());
  return results;
}

}  // namespace interpreter
}  // namespace stablehlo
}  // namespace mlir
