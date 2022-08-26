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
#include <vector>

#include "dialect/StablehloOps.h"
#include "gtest/gtest.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"
#include "mlir/Parser/Parser.h"
#include "reference/Interpreter.h"
#include "reference/Tensor.h"

namespace mlir {
namespace stablehlo {

void runTestCase(
    StringRef sourcePgm,
    ArrayRef<ArrayRef<StringRef>> operandsAndexpectedResultValues) {
  DialectRegistry registry;
  registry.insert<func::FuncDialect>();
  registry.insert<stablehlo::StablehloDialect>();
  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  // Parse the op under test.
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(sourcePgm, &context);
  assert(module && "module parse error!");

  auto main = module->lookupSymbol<mlir::func::FuncOp>("main");
  assert(main && "requires module to have main function");
  auto mainFn = mlir::cast<func::FuncOp>(main);

  auto &op = mainFn.front().front();
  auto numOperands = op.getNumOperands();
  auto numResults = op.getNumResults();

  // Parse operand values.
  SmallVector<Tensor> operandValues;
  for (size_t argIndex = 0; argIndex < numOperands; argIndex++)
    operandValues.push_back(
        makeTensor(op.getOperands()[argIndex].getType(),
                   operandsAndexpectedResultValues[argIndex]));

  // Parse expected values.
  SmallVector<Tensor> expectedResultValues;
  for (size_t resultIdx = 0; resultIdx < numResults; resultIdx++)
    expectedResultValues.push_back(
        makeTensor(op.getResultTypes()[resultIdx],
                   operandsAndexpectedResultValues[numOperands + resultIdx]));

  // Run the test model.
  auto results = eval(mainFn, operandValues);
  ASSERT_TRUE((bool)results) << toString(results.takeError());

  // Check results.
  ASSERT_EQ(results->size(), numResults);
  for (size_t resultIdx = 0; resultIdx < numResults; ++resultIdx) {
    auto expectedType = op.getResultTypes()[resultIdx].cast<ShapedType>();
    ASSERT_EQ((*results)[resultIdx].getType(), expectedType);
    for (int elemIdx = 0; elemIdx < expectedType.getNumElements(); ++elemIdx) {
      ASSERT_EQ((*results)[resultIdx].get(elemIdx),
                expectedResultValues[resultIdx].get(elemIdx));
    }
  }
}

}  // namespace stablehlo
}  // namespace mlir
