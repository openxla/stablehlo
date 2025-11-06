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

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/TypeID.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/transforms/Passes.h"

namespace mlir {
namespace stablehlo {

#define GEN_PASS_DEF_STABLEHLOWRAPINCOMPOSITEPASS
#include "stablehlo/transforms/Passes.h.inc"

namespace {

// Builds a new function within the given `module` that encapsulates the
// functionality of the provided `implOp`. The new function is named uniquely
// and is set to private visibility.
func::FuncOp buildFuncOpWrappingOperation(Operation* op, ModuleOp module) {
  mlir::SymbolTable symbol_table(module);
  // Create an OpBuilder, insertion point at the end of module's body.
  mlir::OpBuilder builder(module);
  builder.setInsertionPointToEnd(&module.getBodyRegion().back());

  // Create the function operation, set private and add to the symbol table.
  // SymbolTable will resolve all name conflicts.
  Location loc = op->getLoc();
  auto funcName = (op->getName().getStringRef() + ".impl").str();
  mlir::func::FuncOp implFunc = mlir::func::FuncOp::create(
      builder, loc, funcName,
      builder.getFunctionType(op->getOperandTypes(), op->getResultTypes()));
  implFunc.setPrivate();
  symbol_table.insert(implFunc);

  Block* block = implFunc.addEntryBlock();
  builder.setInsertionPointToEnd(block);
  mlir::Operation* clonedOp = builder.clone(*op);
  clonedOp->setOperands(block->getArguments());
  mlir::func::ReturnOp::create(builder, loc, clonedOp->getResults());

  return implFunc;
}

// Returns true if the given operation should be wrapped in a CompositeOp.
bool shouldWrapInComposite(
    Operation* op,
    const CompositeAttributeProviderMap& compositeAttributeProviderMap) {
  auto it = compositeAttributeProviderMap.find(op->getName().getTypeID());
  return it != compositeAttributeProviderMap.end() &&
         it->second(op) != std::nullopt;
}

// A ConversionPattern that matches any operation and rewrites it as a
// stablehlo::CompositeOp. The original operation's functionality is
// encapsulated within a newly created private function.
class ConvertGenericOp : public RewritePattern {
 public:
  explicit ConvertGenericOp(
      MLIRContext* context,
      CompositeAttributeProviderMap compositeAttributeProviderMap,
      int32_t compositeVersion)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context),
        compositeAttributeProviderMap(compositeAttributeProviderMap),
        compositeVersion(compositeVersion) {}

  LogicalResult matchAndRewrite(Operation* op,
                                PatternRewriter& rewriter) const override {
    if (!shouldWrapInComposite(op, compositeAttributeProviderMap)) {
      return failure();
    }

    auto module = op->getParentOfType<ModuleOp>();
    if (module == nullptr) {
      return rewriter.notifyMatchFailure(op, "Failed to find enclosing module");
    }

    func::FuncOp decomposition = buildFuncOpWrappingOperation(op, module);
    auto compositeName = op->getName().getStringRef();

    auto compositeAttributeProvider =
        compositeAttributeProviderMap.at(op->getName().getTypeID());
    auto namedAttributes = compositeAttributeProvider(op);
    auto compositeAttributes = rewriter.getDictionaryAttr(*namedAttributes);

    auto compositeOp = stablehlo::CompositeOp::create(
        rewriter, op->getLoc(), op->getResultTypes(), op->getOperands(),
        compositeName, compositeAttributes, decomposition.getSymName(),
        compositeVersion);
    rewriter.replaceOp(op, compositeOp.getResults());
    return success();
  }

 private:
  CompositeAttributeProviderMap compositeAttributeProviderMap;
  int32_t compositeVersion;
};

class StablehloWrapInCompositePass
    : public impl::StablehloWrapInCompositePassBase<
          StablehloWrapInCompositePass> {
 public:
  void initializeAttributeProviderMap(MLIRContext* context,
                                      ArrayRef<std::string> opNames) {
    for (const auto& opNameStr : opNames) {
      StringRef opName = StringRef(opNameStr).trim();

      OperationName registeredOpName = OperationName(opName, context);
      if (!registeredOpName.isRegistered()) {
        llvm::errs() << "Warning: Unknown op name in opNamesOption: '" << opName
                     << "' is not a registered operation.\n";
        continue;
      }

      mlir::TypeID opTypeID = registeredOpName.getTypeID();

      // Create a provider that returns the op's attributes.
      CompositeAttributeProvider provider =
          [](Operation* op) -> std::optional<NamedAttrList> {
        return NamedAttrList(op->getAttrs());
      };
      compositeAttributeProviderMap[opTypeID] = provider;
    }
  }

  StablehloWrapInCompositePass()
      : StablehloWrapInCompositePassBase<StablehloWrapInCompositePass>() {}
  StablehloWrapInCompositePass(const StablehloWrapInCompositePassOptions& opts)
      : StablehloWrapInCompositePassBase<StablehloWrapInCompositePass>(opts) {}
  explicit StablehloWrapInCompositePass(
      const CompositeAttributeProviderMap& compositeAttributeProviderMap,
      int32_t compositeVersion)
      : StablehloWrapInCompositePassBase<StablehloWrapInCompositePass>(),
        compositeAttributeProviderMap(compositeAttributeProviderMap),
        compositeVersion(compositeVersion) {}

  LogicalResult initialize(MLIRContext* context) override {
    RewritePatternSet patterns_(context);
    initializeAttributeProviderMap(context, opNamesOption);
    compositeVersion = versionOption;
    patterns_.add<ConvertGenericOp>(context, compositeAttributeProviderMap,
                                    compositeVersion);
    patterns = std::move(patterns_);
    return success();
  }

  void runOnOperation() override {
    GreedyRewriteConfig config;
    config.setStrictness(GreedyRewriteStrictness::ExistingOps);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns),
                                     config))) {
      signalPassFailure();
      return;
    }
  }

 private:
  FrozenRewritePatternSet patterns;
  CompositeAttributeProviderMap compositeAttributeProviderMap;
  int32_t compositeVersion;
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> createStablehloWrapInCompositePass(
    const CompositeAttributeProviderMap& compositeAttributeProviderMap,
    int32_t compositeVersion) {
  return std::make_unique<StablehloWrapInCompositePass>(
      compositeAttributeProviderMap, compositeVersion);
}

stablehlo::CompositeOp wrapOperationInComposite(OpBuilder& builder,
                                                Operation* op,
                                                const NamedAttrList& attrs,
                                                int32_t compositeVersion,
                                                ModuleOp module) {
  func::FuncOp decomposition = buildFuncOpWrappingOperation(op, module);
  auto compositeName = op->getName().getStringRef();
  auto compositeAttributes = builder.getDictionaryAttr(attrs);
  auto compositeDecomposition = decomposition.getSymName();
  auto composite = stablehlo::CompositeOp::create(
      builder, op->getLoc(), op->getResultTypes(), op->getOperands(),
      compositeName, compositeAttributes, compositeDecomposition,
      compositeVersion);
  return composite;
}

}  // namespace stablehlo
}  // namespace mlir
