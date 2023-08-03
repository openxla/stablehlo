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

#include "llvm/Support/ThreadPool.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
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

LogicalResult RunParallelOp::verify() {
  if (getPrograms().empty() || getPrograms()[0].cast<ArrayAttr>().empty())
    return emitOptionalError(getLoc(), "`programs` attribute cannot be empty");

  size_t numArgs = 0;
  auto numPartitions = getPrograms()[0].cast<ArrayAttr>().size();
  for (auto &replica : getPrograms()) {
    if (replica.cast<ArrayAttr>().size() != numPartitions)
      return emitOptionalError(
          getLoc(), "Sizes of second dimension of `programs` should all match ",
          numPartitions, " but got ", replica.cast<ArrayAttr>().size());

    for (auto &program : replica.cast<ArrayAttr>()) {
      auto funcName = program.cast<StringAttr>();
      auto func =
          (*this)->getParentOfType<ModuleOp>().lookupSymbol<func::FuncOp>(
              funcName);
      if (!func)
        return emitOptionalError(getLoc(), "Function ", funcName, " not found");

      numArgs += func.getNumArguments();
    }
  }

  if (getInputs().size() != numArgs)
    return emitOptionalError(getLoc(),
                             "Number of inputs should match the sum of the "
                             "number of inputs of all programs (",
                             numArgs, ") but got ", getInputs().size());

  return success();
}

//===----------------------------------------------------------------------===//
// Interpreter Ops Evaluator
//===----------------------------------------------------------------------===//

SmallVector<InterpreterValue> evalRunParallelOp(
    ArrayRef<InterpreterValue> inputs,
    SmallVector<SmallVector<StringAttr>> programs, SymbolTable &symbolTable) {
  llvm::ThreadPool threadPool;
  SmallVector<std::shared_future<SmallVector<InterpreterValue>>> futures;
  for (uint32_t i = 0; i < programs.size(); ++i) {
    for (uint32_t j = 0; j < programs[0].size(); ++j) {
      auto funcName = programs[i][j];
      auto func = llvm::cast<func::FuncOp>(symbolTable.lookup(funcName));
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
