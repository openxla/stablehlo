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
#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/DebugStringHelper.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/reference/Errors.h"
#include "stablehlo/reference/InterpreterScope.h"
#include "stablehlo/reference/Ops.h"

namespace mlir {
namespace stablehlo {

namespace {

Tensor fetchOperandInScope(Value value, const InterpreterScope &scope) {
  return scope.find(value);
}

SmallVector<Tensor> fetchVariadicOperandInScope(OperandRange values,
                                                const InterpreterScope &scope) {
  return llvm::to_vector(llvm::map_range(
      values, [&](Value value) { return fetchOperandInScope(value, scope); }));
}

void addOpResultsToScope(Operation &op, ArrayRef<Tensor> runtimeResults,
                         InterpreterScope &scope) {
  assert(op.getNumResults() == runtimeResults.size());
  for (auto [ssaResult, runtimeResult] :
       llvm::zip(op.getResults(), runtimeResults)) {
    if (ssaResult.getType() != runtimeResult.getType()) {
      llvm::errs() << ssaResult.getType() << " " << runtimeResult.getType()
                   << "\n";
      assert(ssaResult.getType() == runtimeResult.getType());
    }
    scope.add(ssaResult, runtimeResult);
  }
}

}  // namespace

llvm::Expected<SmallVector<Tensor>> eval(func::FuncOp func,
                                         ArrayRef<Tensor> args) {
  if (func->getNumRegions() != 1)
    return invalidArgument("Expected one region in func %s",
                           func.getName().str().c_str());

  Block &block = func.front();
  if (block.getNumArguments() != args.size())
    return invalidArgument(
        "Expected same amount of func arguments in %s "
        "and runtime arguments (%d)",
        func.getName().str().c_str(), args.size());

  return eval(func.getBody(), args, nullptr);
}

llvm::Expected<SmallVector<Tensor>> eval(
    Region &region, ArrayRef<Tensor> runtimeArgs,
    const InterpreterScope *const parentScope) {
  if (!region.hasOneBlock())
    return invalidArgument("Expected single block region");

  InterpreterScope scope(parentScope);

  Block &block = region.front();
  if (block.getArguments().size() != runtimeArgs.size())
    report_fatal_error(invalidArgument(
        "Expected same amount of block arguments and runtime arguments (%d)",
        runtimeArgs.size()));

  for (auto [ssaArg, runtimeArg] :
       llvm::zip(block.getArguments(), runtimeArgs)) {
    assert(ssaArg.getType() == runtimeArg.getType());
    scope.add(ssaArg, runtimeArg);
  }

  for (Operation &op : block) {
    if (auto addOp = dyn_cast<AddOp>(op)) {
      Tensor runtimeLhs = fetchOperandInScope(addOp.getLhs(), scope);
      Tensor runtimeRhs = fetchOperandInScope(addOp.getRhs(), scope);
      Tensor runtimeResult = evalAddOp(runtimeLhs, runtimeRhs, addOp.getType());
      addOpResultsToScope(op, {runtimeResult}, scope);
    } else if (auto andOp = dyn_cast<AndOp>(op)) {
      Tensor runtimeLhs = fetchOperandInScope(andOp.getLhs(), scope);
      Tensor runtimeRhs = fetchOperandInScope(andOp.getRhs(), scope);
      Tensor runtimeResult = evalAndOp(runtimeLhs, runtimeRhs, andOp.getType());
      addOpResultsToScope(op, {runtimeResult}, scope);
    } else if (auto ceilOp = dyn_cast<CeilOp>(op)) {
      Tensor runtimeOperand = fetchOperandInScope(ceilOp.getOperand(), scope);
      Tensor runtimeResult = evalCeilOp(runtimeOperand, ceilOp.getType());
      addOpResultsToScope(op, {runtimeResult}, scope);
    } else if (auto constantOp = dyn_cast<ConstantOp>(op)) {
      Tensor runtimeResult = evalConstantOp(constantOp.getValue());
      addOpResultsToScope(op, {runtimeResult}, scope);
    } else if (auto cosineOp = dyn_cast<CosineOp>(op)) {
      Tensor runtimeOperand = fetchOperandInScope(cosineOp.getOperand(), scope);
      Tensor runtimeResult = evalCosineOp(runtimeOperand, cosineOp.getType());
      addOpResultsToScope(op, {runtimeResult}, scope);
    } else if (auto floorOp = dyn_cast<FloorOp>(op)) {
      Tensor runtimeOperand = fetchOperandInScope(floorOp.getOperand(), scope);
      Tensor runtimeResult = evalFloorOp(runtimeOperand, floorOp.getType());
      addOpResultsToScope(op, {runtimeResult}, scope);
    } else if (auto iotaOp = dyn_cast<IotaOp>(op)) {
      Tensor runtimeResult =
          evalIotaOp(iotaOp.getIotaDimension(), iotaOp.getType());
      addOpResultsToScope(op, {runtimeResult}, scope);
    } else if (auto maxOp = dyn_cast<MaxOp>(op)) {
      Tensor runtimeLhs = fetchOperandInScope(maxOp.getLhs(), scope);
      Tensor runtimeRhs = fetchOperandInScope(maxOp.getRhs(), scope);
      Tensor runtimeResult = evalMaxOp(runtimeLhs, runtimeRhs, maxOp.getType());
      addOpResultsToScope(op, {runtimeResult}, scope);
    } else if (auto minOp = dyn_cast<MinOp>(op)) {
      Tensor runtimeLhs = fetchOperandInScope(minOp.getLhs(), scope);
      Tensor runtimeRhs = fetchOperandInScope(minOp.getRhs(), scope);
      Tensor runtimeResult = evalMinOp(runtimeLhs, runtimeRhs, minOp.getType());
      addOpResultsToScope(op, {runtimeResult}, scope);
    } else if (auto multiplyOp = dyn_cast<MulOp>(op)) {
      Tensor runtimeLhs = fetchOperandInScope(multiplyOp.getLhs(), scope);
      Tensor runtimeRhs = fetchOperandInScope(multiplyOp.getRhs(), scope);
      Tensor runtimeResult =
          evalMultiplyOp(runtimeLhs, runtimeRhs, multiplyOp.getType());
      addOpResultsToScope(op, {runtimeResult}, scope);
    } else if (auto negOp = dyn_cast<NegOp>(op)) {
      Tensor runtimeOperand = fetchOperandInScope(negOp.getOperand(), scope);
      Tensor runtimeResult = evalNegOp(runtimeOperand, negOp.getType());
      addOpResultsToScope(op, {runtimeResult}, scope);
    } else if (auto notOp = dyn_cast<NotOp>(op)) {
      Tensor runtimeOperand = fetchOperandInScope(notOp.getOperand(), scope);
      Tensor runtimeResult = evalNotOp(runtimeOperand, notOp.getType());
      addOpResultsToScope(op, {runtimeResult}, scope);
    } else if (auto orOp = dyn_cast<OrOp>(op)) {
      Tensor runtimeLhs = fetchOperandInScope(orOp.getLhs(), scope);
      Tensor runtimeRhs = fetchOperandInScope(orOp.getRhs(), scope);
      Tensor runtimeResult = evalOrOp(runtimeLhs, runtimeRhs, orOp.getType());
      addOpResultsToScope(op, {runtimeResult}, scope);
    } else if (auto whileOp = dyn_cast<WhileOp>(op)) {
      auto runtimeInputs =
          fetchVariadicOperandInScope(whileOp.getOperand(), scope);
      auto runtimeResults = evalWhileOp(runtimeInputs, whileOp.getCond(),
                                        whileOp.getBody(), scope);
      addOpResultsToScope(op, runtimeResults, scope);
    } else if (auto reshapeOp = dyn_cast<ReshapeOp>(op)) {
      Tensor runtimeOperand =
          fetchOperandInScope(reshapeOp.getOperand(), scope);
      Tensor runtimeResult = evalReshapeOp(runtimeOperand, reshapeOp.getType());
      addOpResultsToScope(op, {runtimeResult}, scope);
    } else if (auto reverseOp = dyn_cast<ReverseOp>(op)) {
      Tensor runtimeOperand =
          fetchOperandInScope(reverseOp.getOperand(), scope);
      auto dimensions =
          llvm::to_vector(reverseOp.getDimensions().getValues<int64_t>());
      Tensor runtimeResult =
          evalReverseOp(runtimeOperand, dimensions, reverseOp.getType());
      addOpResultsToScope(op, {runtimeResult}, scope);
    } else if (auto returnOp = dyn_cast<func::ReturnOp>(op)) {
      SmallVector<Tensor> runtimeOperands;
      for (Value ssaOperand : returnOp.getOperands())
        runtimeOperands.push_back(fetchOperandInScope(ssaOperand, scope));
      return runtimeOperands;
    } else if (auto returnOp = dyn_cast<ReturnOp>(op)) {
      SmallVector<Tensor> runtimeOperands;
      for (Value ssaOperand : returnOp.getOperands())
        runtimeOperands.push_back(fetchOperandInScope(ssaOperand, scope));
      return runtimeOperands;
    } else if (auto sineOp = dyn_cast<SineOp>(op)) {
      Tensor runtimeOperand = fetchOperandInScope(sineOp.getOperand(), scope);
      Tensor runtimeResult = evalSineOp(runtimeOperand, sineOp.getType());
      addOpResultsToScope(op, {runtimeResult}, scope);
    } else if (auto sliceOp = dyn_cast<SliceOp>(op)) {
      Tensor runtimeOperand = fetchOperandInScope(sliceOp.getOperand(), scope);
      auto startIndices =
          llvm::to_vector(sliceOp.getStartIndices().getValues<int64_t>());
      auto strides = llvm::to_vector(sliceOp.getStrides().getValues<int64_t>());
      Tensor runtimeResult =
          evalSliceOp(runtimeOperand, startIndices, strides, sliceOp.getType());
      addOpResultsToScope(op, {runtimeResult}, scope);
    } else if (auto subtractOp = dyn_cast<SubtractOp>(op)) {
      Tensor runtimeLhs = fetchOperandInScope(subtractOp.getLhs(), scope);
      Tensor runtimeRhs = fetchOperandInScope(subtractOp.getRhs(), scope);
      Tensor runtimeResult =
          evalSubtractOp(runtimeLhs, runtimeRhs, subtractOp.getType());
      addOpResultsToScope(op, {runtimeResult}, scope);
    } else if (auto tanhOp = dyn_cast<TanhOp>(op)) {
      Tensor runtimeOperand = fetchOperandInScope(tanhOp.getOperand(), scope);
      Tensor runtimeResult = evalTanhOp(runtimeOperand, tanhOp.getType());
      addOpResultsToScope(op, {runtimeResult}, scope);
    } else if (auto transposeOp = dyn_cast<TransposeOp>(op)) {
      Tensor runtimeOperand =
          fetchOperandInScope(transposeOp.getOperand(), scope);
      auto permutation =
          llvm::to_vector(transposeOp.getPermutation().getValues<int64_t>());
      Tensor runtimeResult =
          evalTransposeOp(runtimeOperand, permutation, transposeOp.getType());
      addOpResultsToScope(op, {runtimeResult}, scope);
    } else if (auto xorOp = dyn_cast<XorOp>(op)) {
      Tensor runtimeLhs = fetchOperandInScope(xorOp.getLhs(), scope);
      Tensor runtimeRhs = fetchOperandInScope(xorOp.getRhs(), scope);
      Tensor runtimeResult = evalXorOp(runtimeLhs, runtimeRhs, xorOp.getType());
      addOpResultsToScope(op, {runtimeResult}, scope);
    } else {
      return invalidArgument("Unsupported op: %s", debugString(op).c_str());
    }
  }

  return invalidArgument("Expected a terminator when evaluating func");
}

}  // namespace stablehlo
}  // namespace mlir
