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

#include "Interpreter.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Errc.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/DebugStringHelper.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/reference/Ops.h"

namespace mlir {
namespace stablehlo {

namespace {

template <typename... Ts>
inline llvm::Error invalidArgument(char const *Fmt, const Ts &...Vals) {
  return createStringError(llvm::errc::invalid_argument, Fmt, Vals...);
}

}  // namespace

llvm::Expected<SmallVector<Tensor>> eval(func::FuncOp func,
                                         ArrayRef<Tensor> args) {
  if (func->getNumRegions() != 1) {
    return invalidArgument("Expected one region in func %s",
                           func.getName().str().c_str());
  }
  if (!func.getBody().hasOneBlock()) {
    return invalidArgument("Expected one block in func %s",
                           func.getName().str().c_str());
  }

  Block &block = func.front();
  if (block.getNumArguments() != args.size()) {
    return invalidArgument(
        "Expected same amount of func arguments in %s "
        "and runtime arguments (%d)",
        func.getName().str().c_str(), args.size());
  }
  llvm::DenseMap<Value, Tensor> stackFrame;
  for (auto [ssaArg, runtimeArg] : llvm::zip(block.getArguments(), args)) {
    stackFrame[ssaArg] = runtimeArg;
  }

  for (Operation &op : block) {
    auto fetchOperand = [&](Value value) -> Tensor {
      auto it = stackFrame.find(value);
      if (it != stackFrame.end()) return it->second;

      auto err = invalidArgument("Expected a terminator when evaluating func");
      report_fatal_error(std::move(err));
    };
    auto populateResults = [&](ArrayRef<Tensor> runtimeValues) {
      assert(op.getNumResults() == runtimeValues.size());
      for (auto [ssaResult, runtimeResult] :
           llvm::zip(op.getResults(), runtimeValues)) {
        stackFrame[ssaResult] = runtimeResult;
      }
    };

    if (auto addOp = dyn_cast<AddOp>(op)) {
      Tensor runtimeLhs = fetchOperand(addOp.lhs());
      Tensor runtimeRhs = fetchOperand(addOp.rhs());
      Tensor runtimeResult = eval(addOp, runtimeLhs, runtimeRhs);
      populateResults({runtimeResult});
    } else if (auto constantOp = dyn_cast<ConstantOp>(op)) {
      Tensor runtimeResult = eval(constantOp, constantOp.value());
      populateResults({runtimeResult});
    } else if (auto returnOp = dyn_cast<func::ReturnOp>(op)) {
      SmallVector<Tensor> runtimeOperands;
      for (Value ssaOperand : returnOp.operands()) {
        runtimeOperands.push_back(fetchOperand(ssaOperand));
      }
      return runtimeOperands;
    } else {
      return invalidArgument("Unsupported op: %s", debugString(op).c_str());
    }
  }

  return invalidArgument("Expected a terminator when evaluating func");
}

}  // namespace stablehlo
}  // namespace mlir
