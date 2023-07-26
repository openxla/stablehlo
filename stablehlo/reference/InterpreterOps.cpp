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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/reference/InterpreterValue.h"
#include "stablehlo/reference/Ops.h"
#include "stablehlo/reference/ProcessGrid.h"

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

SmallVector<InterpreterValue> evalRunParallelOp(
    ArrayRef<InterpreterValue> inputs, ArrayRef<StringRef> programs,
    uint32_t numReplicas, uint32_t numPartitions, Operation &op) {
  std::map<ProcessId, std::future<SmallVector<InterpreterValue>>> futureResults;
  std::map<ProcessId, std::thread> processMap;
  ProcessGrid processGrid{};
  SmallVector<InterpreterValue> results;
  for (uint32_t i = 0; i < numReplicas; ++i) {
    for (uint32_t j = 0; j < numPartitions; ++j) {
      ProcessId processId{i, j};

      auto evalWrapper = [](Region &region, ArrayRef<InterpreterValue> args,
                            ProcessId processId, ProcessGrid *processGrid) {
        Process process{processId, processGrid};
        return eval(region, args, &process, /*parent=*/nullptr,
                    /*fallback=*/nullptr);
      };
      std::packaged_task<SmallVector<InterpreterValue>(
          Region &, ArrayRef<InterpreterValue>, ProcessId, ProcessGrid *)>
          task(evalWrapper);

      auto &parentBlock =
          op.getParentRegion()->getParentRegion()->getBlocks().front();
      auto program = std::find_if(
          parentBlock.begin(), parentBlock.end(), [&](Operation &op) {
            return dyn_cast<func::FuncOp>(op).getSymName().equals(programs[j]);
          });

      futureResults[processId] = task.get_future();
      processMap[processId] = std::thread(
          std::move(task), std::ref(dyn_cast<func::FuncOp>(program).getBody()),
          inputs, processId, &processGrid);
    }
  }

  for (uint32_t i = 0; i < numReplicas; ++i) {
    for (uint32_t j = 0; j < numPartitions; ++j) {
      ProcessId processId{i, j};
      processMap[processId].join();
    }
  }

  for (uint32_t i = 0; i < numReplicas; ++i) {
    for (uint32_t j = 0; j < numPartitions; ++j) {
      ProcessId processId{i, j};
      results.append(futureResults[processId].get());
    }
  }
  return results;
}

}  // namespace interpreter
}  // namespace stablehlo
}  // namespace mlir
