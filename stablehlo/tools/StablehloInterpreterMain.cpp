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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/reference/Errors.h"
#include "stablehlo/reference/Ops.h"
#include "stablehlo/reference/Scope.h"
#include "stablehlo/reference/Tensor.h"
#include "stablehlo/tests/CheckOps.h"

namespace mlir {

namespace {

stablehlo::Tensor makeBooleanTensor(MLIRContext *context, bool value) {
  auto builder = Builder(context);
  auto type = RankedTensorType::get({}, builder.getI1Type());
  auto res = DenseElementsAttr::get(type, builder.getBoolAttr(true));
  return stablehlo::makeTensor(res);
}

llvm::Error evalCustomCallCheckEq(stablehlo::CustomCallOp op,
                                  stablehlo::Scope &scope,
                                  llvm::StringRef funcName) {
  if (op->getNumOperands() != 2) {
    return stablehlo::invalidArgument("Unsupported op: %s",
                                      debugString(op).c_str());
  }
  stablehlo::Tensor actualResult = scope.find(op->getOperands())[0];
  auto expectedOp =
      dyn_cast<stablehlo::ConstantOp>(op->getOperands()[1].getDefiningOp());

  // TODO: Need to handle complex types.
  bool isInt = expectedOp.getType().getElementType().isa<IntegerType>();
  auto expectedResult = expectedOp.getValue();
  auto status =
      isInt ? stablehlo::check::evalEqOp(actualResult, expectedResult)
            : stablehlo::check::evalAlmostEqOp(actualResult, expectedResult);
  if (status)
    return stablehlo::invalidArgument(
        "Error evaluating function: %s. \n\tCheck almost_eq failed: "
        "%s",
        funcName.data(), toString(std::move(status)).c_str());

  scope.add(op.getResults(), makeBooleanTensor(op->getContext(), true));
  return llvm::Error::success();
}

llvm::Error interpreterFallback(Operation &op, stablehlo::Scope &scope,
                                llvm::StringRef funcName) {
  // TODO: Write a pass to inline this custom call into a check op.
  if (auto customCall = dyn_cast<stablehlo::CustomCallOp>(op)) {
    if (customCall.getCallTargetName() == "check.eq")
      return evalCustomCallCheckEq(customCall, scope, funcName);
    return stablehlo::invalidArgument("Unsupported custom call: %s",
                                      debugString(op).c_str());
  }

  // check.almost_eq implementation
  if (auto almostEqOp = dyn_cast<stablehlo::check::AlmostEqOp>(op)) {
    stablehlo::Tensor runtimeOperand = scope.find(almostEqOp.getLhs());
    auto status =
        stablehlo::check::evalAlmostEqOp(runtimeOperand, almostEqOp.getValue());
    if (status)
      return stablehlo::invalidArgument(
          "Error evaluating function: %s. \n\tCheck almost_eq failed: "
          "%s",
          funcName.data(), toString(std::move(status)).c_str());
    return llvm::Error::success();
  }

  // check.eq implementation
  if (auto eqOp = dyn_cast<stablehlo::check::EqOp>(op)) {
    stablehlo::Tensor runtimeOperand = scope.find(eqOp.getLhs());
    auto status = stablehlo::check::evalEqOp(runtimeOperand, eqOp.getValue());
    if (status)
      return stablehlo::invalidArgument(
          "Error evaluating function: %s. \n\tCheck eq failed: %s",
          funcName.data(), toString(std::move(status)).c_str());
    return llvm::Error::success();
  }
  return stablehlo::invalidArgument("Unsupported op: %s",
                                    debugString(op).c_str());
}
}  // namespace

TranslateFromMLIRRegistration stablehlo_interpreter(
    "interpret", "Interpreter for StableHLO",
    [](ModuleOp module, raw_ostream &os) {
      auto walkResult = module.walk([&](func::FuncOp funcOp) {
        auto interpreterFallbackFn = [&](Operation &op,
                                         stablehlo::Scope &scope) {
          return interpreterFallback(op, scope, funcOp.getSymName());
        };

        // Run the test model.
        auto results = stablehlo::eval(funcOp.getBody(), {}, /*parent=*/nullptr,
                                       interpreterFallbackFn);

        // Dump the results.
        for (auto &result : results) result.print(os);
        return WalkResult::advance();
      });

      return success(!walkResult.wasInterrupted());
    },
    [](DialectRegistry &registry) {
      registry.insert<func::FuncDialect>();
      registry.insert<stablehlo::check::CheckDialect>();
      registry.insert<stablehlo::StablehloDialect>();
    });

}  //  namespace mlir

int main(int argc, char **argv) {
  return failed(
      mlir::mlirTranslateMain(argc, argv, "StableHLO interpreter driver\n"));
}
