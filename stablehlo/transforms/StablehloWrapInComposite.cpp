/* Copyright 2024 The StableHLO Authors. All Rights Reserved.

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
#include <memory>
#include <string>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/transforms/Passes.h"

namespace mlir {
namespace stablehlo {

#define GEN_PASS_DEF_STABLEHLOWRAPINCOMPOSITEPASS
#include "stablehlo/transforms/Passes.h.inc"

namespace {

// Generates a unique function name based on the given `baseFuncName` within
// the provided `module`. Ensures the generated name does not clash with any
// existing symbols by appending a counter if necessary.
std::string generateUniqueFunctionName(StringRef baseFuncName,
                                       mlir::ModuleOp module) {
  mlir::SymbolTable symbolTable(module);
  int counter = 0;
  std::string baseName = baseFuncName.str() + ".impl";
  std::string funcName = baseName;
  while (symbolTable.lookup(funcName)) {
    counter++;
    funcName = (baseName + std::to_string(counter));
  }
  return funcName;
}

// Builds a new function within the given `module` that encapsulates the
// functionality of the provided `implOp`. The new function is named uniquely
// and is set to private visibility.
mlir::func::FuncOp buildStableHLOCompositeImplFunc(mlir::ModuleOp module,
                                                   mlir::Operation* implOp) {
  // Create an OpBuilder, insertion point at the end of module's body.
  mlir::OpBuilder builder(module);
  builder.setInsertionPointToEnd(&module.getBodyRegion().back());

  // Prepare argument types and locations for the new function.
  llvm::SmallVector<mlir::Location> argLocs;
  llvm::SmallVector<mlir::Type> argTypes;
  for (auto& operand : implOp->getOpOperands()) {
    argTypes.push_back(
        operand.get().getType());  // Get the type of each operand.
    argLocs.push_back(
        operand.get().getLoc());  // Get the location of each operand.
  }

  // Prepare result types for the new function.
  llvm::SmallVector<mlir::Type> resultTypes;
  for (auto result : implOp->getResults()) {
    resultTypes.push_back(result.getType());  // Get the type of each result.
  }

  // Create the function operation.
  auto uniqueFuncName =
      generateUniqueFunctionName(implOp->getName().getStringRef(), module);
  mlir::func::FuncOp implFunc = builder.create<mlir::func::FuncOp>(
      module.getLoc(), uniqueFuncName,
      builder.getFunctionType(argTypes,
                              resultTypes));  // Set arg and result types

  // Create a block in the function body representing the function's content
  // and map the arguments from the original op to the new function.
  mlir::IRMapping mapping;  // Maps values from the old op to the new function.
  builder.createBlock(&implFunc.getBody(), implFunc.begin(), argTypes, argLocs);

  // Map the operands of the original op to the arguments of the newly created
  // function.
  for (const auto& operand : llvm::enumerate(implOp->getOperands())) {
    mapping.map(operand.value(), implFunc.getArgument(operand.index()));
  }

  // Clone the original operation into the body of the new function,
  // using the value mapping to remap operands.
  mlir::Operation* cloned_op = builder.clone(*implOp, mapping);

  // Create the return operation, returning the results of the cloned op.
  llvm::SmallVector<mlir::Value> results;
  results.append(cloned_op->getResults().begin(),
                 cloned_op->getResults().end());
  builder.create<mlir::func::ReturnOp>(implFunc.getBody().getLoc(), results);

  // Add the newly created function to the module's symbol table and make it
  // private.
  mlir::SymbolTable symbol_table(module);
  implFunc.setPrivate();
  symbol_table.insert(implFunc);

  return implFunc;
}

// A ConversionPattern that matches any operation and rewrites it as a
// stablehlo::CompositeOp. The original operation's functionality is
// encapsulated within a newly created private function.
class ConvertGenericOp : public ConversionPattern {
 public:
  explicit ConvertGenericOp(MLIRContext* ctx)
      : ConversionPattern(MatchAnyOpTypeTag(), /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(
      Operation* op, ArrayRef<Value> operands,
      ConversionPatternRewriter& rewriter) const override {
    // get the enclosing module
    auto module = op->getParentOfType<ModuleOp>();
    MLIRContext* context = op->getContext();
    if (module == nullptr) {
      return rewriter.notifyMatchFailure(op, "Failed to find enclosing module");
    }
    auto implFunc = buildStableHLOCompositeImplFunc(module, op);
    auto name = op->getName().getStringRef();

    llvm::SmallVector<mlir::Value> compositeOperands(op->operand_begin(),
                                                     op->operand_end());
    auto compositeOp = rewriter.create<stablehlo::CompositeOp>(
        op->getLoc(), op->getResultTypes(), compositeOperands, name,
        DictionaryAttr::get(context, op->getAttrs()), implFunc.getSymName());
    rewriter.replaceOp(op, compositeOp.getResults());
    return success();
  }
};

class StablehloWrapInCompositePass
    : public impl::StablehloWrapInCompositePassBase<
          StablehloWrapInCompositePass> {
 public:
  StablehloWrapInCompositePass()
      : StablehloWrapInCompositePassBase<StablehloWrapInCompositePass>() {}
  StablehloWrapInCompositePass(const StablehloWrapInCompositePassOptions& opts)
      : StablehloWrapInCompositePassBase<StablehloWrapInCompositePass>(opts) {}
  explicit StablehloWrapInCompositePass(
      std::function<bool(Operation*)> opPredicate) {
    this->opPredicate = opPredicate;
  }
  LogicalResult initialize(MLIRContext* context) override {
    RewritePatternSet patterns_(context);
    patterns_.add<ConvertGenericOp>(context);
    patterns = std::move(patterns_);

    if (!opPredicate) {
      if (!opNamesOption.empty()) {
        DenseSet<StringRef> opNames(opNamesOption.begin(), opNamesOption.end());
        opPredicate = [opNames](Operation* op) {
          return opNames.contains(op->getName().getStringRef());
        };
      } else {
        opPredicate = [](Operation* op) { return false; };
      }
    }
    return success();
  }

  void runOnOperation() override {
    ConversionTarget target(getContext());
    target.addLegalDialect<func::FuncDialect>();
    target.addDynamicallyLegalDialect<stablehlo::StablehloDialect>(
        [this](Operation* op) { return !opPredicate(op); });

    Operation* op = getOperation();
    if (failed(applyPartialConversion(op, target, std::move(patterns)))) {
      getOperation().emitError("Wrap in composite pass failed.");
      signalPassFailure();
    }
  }

 private:
  // FrozenRewritePatternSet for the pass.
  FrozenRewritePatternSet patterns;
  // Predicate function to determine which operations should be wrapped.
  std::function<bool(Operation*)> opPredicate = nullptr;
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> createStablehloWrapInCompositePass(
    std::function<bool(Operation*)> opPredicate) {
  return std::make_unique<StablehloWrapInCompositePass>(opPredicate);
}

}  // namespace stablehlo
}  // namespace mlir
