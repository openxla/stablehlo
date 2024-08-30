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

#include <fcntl.h>

#include <cassert>

#include "llvm/ADT/APFloat.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/dialect/Version.h"
#include "stablehlo/transforms/Passes.h"

namespace mlir {
namespace stablehlo {
#define GEN_PASS_DEF_STABLEHLOCREATECOMPATIBILITYEXPANDERPASS
#include "stablehlo/transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Helpers.
//===----------------------------------------------------------------------===//

// Creates a constant with all ones.
static Value createConstantWithAllOnes(OpBuilder &b, Location loc, Value val) {
  if (!isa<FloatType>(getElementTypeOrSelf(val)))
    llvm_unreachable("Unsupported element type, expecting float");

  auto shapedTy = dyn_cast<mlir::ShapedType>(val.getType());
  if (!shapedTy) llvm_unreachable("Unsupported shaped type.");

  mlir::DenseElementsAttr elementsAttr =
      mlir::DenseElementsAttr::get(shapedTy, 1.0);

  return b.create<mlir::stablehlo::ConstantOp>(loc, val.getType(),
                                               elementsAttr);
}

// Check user-specified target version.
vhlo::Version validateTargetVersion(llvm::StringRef versionRef) {
  auto failOrVersion = vhlo::Version::fromString(versionRef);
  if (failed(failOrVersion)) {
    assert(!versionRef.empty() &&
           "No target version specified. Target version must be of the form "
           "`#.#.#`.");
    assert(versionRef.empty() &&
           "Invalid target version argument. Target version must be of the "
           "form `#.#.#`.");
  }
  vhlo::Version targetVersion = *failOrVersion;
  assert((vhlo::Version::getMinimumVersion() <= targetVersion) &&
         "target version is less than minimum supported.");
  assert((targetVersion <= vhlo::Version::getCurrentVersion()) &&
         "target version is greater than current version.");
  return targetVersion;
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct StablehloCreateCompatibilityExpanderPass
    : public impl::StablehloCreateCompatibilityExpanderPassBase<
          StablehloCreateCompatibilityExpanderPass> {
  StablehloCreateCompatibilityExpanderPass()
      : StablehloCreateCompatibilityExpanderPassBase<
            StablehloCreateCompatibilityExpanderPass>() {}
  StablehloCreateCompatibilityExpanderPass(
      const StablehloCreateCompatibilityExpanderPassOptions &opts)
      : StablehloCreateCompatibilityExpanderPassBase<
            StablehloCreateCompatibilityExpanderPass>(opts) {}

 public:
  LogicalResult initialize(MLIRContext *context) override {
    auto targetVersion = validateTargetVersion(targetVersionOption);

    config.useTopDownTraversal = true;
    RewritePatternSet patterns_(context);
    populateStablehloCreateCompatibilityExpanderPatterns(&patterns_, context,
                                                         targetVersion);
    patterns = std::move(patterns_);
    return success();
  }

  void runOnOperation() override {
    auto func = getOperation();
    if (failed(applyPatternsAndFoldGreedily(func, patterns, config))) {
      func.emitError(
          "Failed to converge StableHLOCreateCompatibilityExpanderPass in ")
          << config.maxIterations << " iterations";
      signalPassFailure();
    }
  }

 private:
  FrozenRewritePatternSet patterns;
  GreedyRewriteConfig config;
};

#include "stablehlo/transforms/StablehloCreateCompatibilityExpanderPatterns.h.inc"

}  // namespace

void populateStablehloCreateCompatibilityExpanderPatterns(
    RewritePatternSet *patterns, MLIRContext *context,
    vhlo::Version targetVersion) {
  // StableHLO TanOp is introduced in v1.4.0.
  if (targetVersion < vhlo::Version(1, 4, 0)) {
    patterns->add<TanOp_ComplexElementType_CompatiblityExpander>(context);
    patterns->add<TanOp_CompatiblityExpander>(context);
  }
}

}  // namespace stablehlo
}  // namespace mlir
