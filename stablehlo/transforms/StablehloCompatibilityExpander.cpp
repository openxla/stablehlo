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

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iterator>
#include <utility>

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/dialect/Version.h"
#include "stablehlo/transforms/PassUtils.h"
#include "stablehlo/transforms/Passes.h"

namespace mlir {
namespace stablehlo {
#define GEN_PASS_DEF_STABLEHLOCOMPATIBILITYEXPANDERPASS
#include "stablehlo/transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Helpers.
//===----------------------------------------------------------------------===//

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

SmallVector<int64_t> mergeSortedDims(ArrayRef<int64_t> dims1,
                                     ArrayRef<int64_t> dims2) {
  SmallVector<int64_t> result;
  result.reserve(dims1.size() + dims2.size());
  std::merge(dims1.begin(), dims1.end(), dims2.begin(), dims2.end(),
             std::back_inserter(result));
  return result;
}

// Returns an updated indices tensor such that an `IotaOp` is prepended for each
// dim in `indicesBatchingDims` with a `ConcatenateOp`.
//
// If `indexVectorDim` is equal to the rank of `indices`, it is reshaped to have
// a trailing dimension of size 1 so it can be concatenated with the `IotaOp`s.
Value createConcatIndices(Value indices, int64_t indexVectorDim,
                          ArrayRef<int64_t> indicesBatchingDims,
                          PatternRewriter &rewriter) {
  Location loc = indices.getLoc();
  auto indicesType = cast<RankedTensorType>(indices.getType());
  bool indexVectorDimOnLastDim = indexVectorDim == indicesType.getRank();

  SmallVector<int64_t> iotaShape(indicesType.getShape());
  if (indexVectorDimOnLastDim) {
    iotaShape.push_back(1);
  } else {
    iotaShape[indexVectorDim] = 1;
  }
  auto iotaType =
      RankedTensorType::get(iotaShape, indicesType.getElementType());

  SmallVector<Value> indicesToConcat;
  indicesToConcat.reserve(indicesBatchingDims.size() + 1);
  for (int64_t batchingDim : indicesBatchingDims) {
    indicesToConcat.push_back(
        rewriter.create<IotaOp>(loc, iotaType, batchingDim));
  }
  if (indexVectorDimOnLastDim) {
    indicesToConcat.push_back(
        rewriter.create<ReshapeOp>(loc, iotaType, indices));
  } else {
    indicesToConcat.push_back(indices);
  }
  return rewriter.create<ConcatenateOp>(loc, indicesToConcat, indexVectorDim);
}

//===----------------------------------------------------------------------===//
// Patterns (non DRR)
//===----------------------------------------------------------------------===//

// Converts a `GatherOp` with batching dims to a `GatherOp` without batching
// dims, such that each batching dim becomes a collapsed slice dim with a
// corresponding `IotaOp` concatenated to the start indices.
class GatherWithBatchingDimsExpander : public OpRewritePattern<GatherOp> {
  using OpRewritePattern<GatherOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GatherOp op,
                                PatternRewriter &rewriter) const override {
    GatherDimensionNumbersAttr dimNumbers = op.getDimensionNumbers();
    ArrayRef<int64_t> operandBatchingDims = dimNumbers.getOperandBatchingDims();
    if (operandBatchingDims.empty())
      return rewriter.notifyMatchFailure(op, [](Diagnostic &diag) {
        diag << "gather op has no batching dims";
      });

    SmallVector<int64_t> newCollapsedSliceDims = mergeSortedDims(
        operandBatchingDims, dimNumbers.getCollapsedSliceDims());
    SmallVector<int64_t> newStartIndexMap =
        llvm::to_vector(llvm::concat<const int64_t>(
            operandBatchingDims, dimNumbers.getStartIndexMap()));
    Value newIndices = createConcatIndices(
        op.getStartIndices(), dimNumbers.getIndexVectorDim(),
        dimNumbers.getStartIndicesBatchingDims(), rewriter);
    rewriter.replaceOpWithNewOp<GatherOp>(
        op, op.getOperand(), newIndices,
        GatherDimensionNumbersAttr::get(
            op.getContext(), dimNumbers.getOffsetDims(), newCollapsedSliceDims,
            /*operandBatchingDims=*/{}, /*startIndicesBatchingDims=*/{},
            newStartIndexMap, dimNumbers.getIndexVectorDim()),
        op.getSliceSizes(), /*indicesAreSorted=*/false);

    return success();
  }
};

// Converts a `ScatterOp` with batching dims to a `ScatterOp` without batching
// dims, such that each batching dim becomes an inserted window dim with a
// corresponding `IotaOp` concatenated to the scatter indices.
class ScatterWithBatchingDimsExpander : public OpRewritePattern<ScatterOp> {
  using OpRewritePattern<ScatterOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ScatterOp op,
                                PatternRewriter &rewriter) const override {
    ScatterDimensionNumbersAttr dimNumbers = op.getScatterDimensionNumbers();
    ArrayRef<int64_t> inputBatchingDims = dimNumbers.getInputBatchingDims();
    if (inputBatchingDims.empty())
      return rewriter.notifyMatchFailure(op, [](Diagnostic &diag) {
        diag << "scatter op has no batching dims";
      });

    SmallVector<int64_t> newInsertedWindowDims =
        mergeSortedDims(inputBatchingDims, dimNumbers.getInsertedWindowDims());
    SmallVector<int64_t> newScatterDimsToOperandDims =
        llvm::to_vector(llvm::concat<const int64_t>(
            inputBatchingDims, dimNumbers.getScatterDimsToOperandDims()));
    Value newIndices = createConcatIndices(
        op.getScatterIndices(), dimNumbers.getIndexVectorDim(),
        dimNumbers.getScatterIndicesBatchingDims(), rewriter);
    auto newScatterOp = rewriter.create<ScatterOp>(
        op.getLoc(), op->getResultTypes(), op.getInputs(), newIndices,
        op.getUpdates(),
        ScatterDimensionNumbersAttr::get(
            op.getContext(), dimNumbers.getUpdateWindowDims(),
            newInsertedWindowDims,
            /*inputBatchingDims=*/{}, /*scatterIndicesBatchingDims=*/{},
            newScatterDimsToOperandDims, dimNumbers.getIndexVectorDim()),
        /*indicesAreSorted=*/false, op.getUniqueIndices());

    newScatterOp.getUpdateComputation().takeBody(op.getUpdateComputation());
    rewriter.replaceOp(op, newScatterOp.getResults());

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct StablehloCompatibilityExpanderPass
    : public impl::StablehloCompatibilityExpanderPassBase<
          StablehloCompatibilityExpanderPass> {
  StablehloCompatibilityExpanderPass()
      : StablehloCompatibilityExpanderPassBase<
            StablehloCompatibilityExpanderPass>() {}
  StablehloCompatibilityExpanderPass(
      const StablehloCompatibilityExpanderPassOptions &opts)
      : StablehloCompatibilityExpanderPassBase<
            StablehloCompatibilityExpanderPass>(opts) {}

 public:
  LogicalResult initialize(MLIRContext *context) override {
    auto targetVersion = validateTargetVersion(targetVersionOption);

    config.useTopDownTraversal = true;
    RewritePatternSet patterns_(context);
    populateStablehloCompatibilityExpanderPatterns(&patterns_, context,
                                                   targetVersion);
    patterns = std::move(patterns_);
    return success();
  }

  void runOnOperation() override {
    auto func = getOperation();
    if (failed(applyPatternsAndFoldGreedily(func, patterns, config))) {
      func.emitError(
          "Failed to converge StableHLOCompatibilityExpanderPass in ")
          << config.maxIterations << " iterations";
      signalPassFailure();
    }
  }

 private:
  FrozenRewritePatternSet patterns;
  GreedyRewriteConfig config;
};

#include "stablehlo/transforms/StablehloCompatibilityExpanderPatterns.h.inc"

}  // namespace

void populateStablehloCompatibilityExpanderPatterns(
    RewritePatternSet *patterns, MLIRContext *context,
    vhlo::Version targetVersion) {
  // StableHLO GatherOp/ScatterOp with batching dims is introduced in v1.1.0.
  if (targetVersion < vhlo::Version(1, 1, 0))
    patterns
        ->add<GatherWithBatchingDimsExpander, ScatterWithBatchingDimsExpander>(
            context);

  // StableHLO TanOp is introduced in v1.4.0.
  if (targetVersion < vhlo::Version(1, 4, 0))
    patterns->add<TanOp_ComplexElementType_CompatiblityExpander,
                  TanOp_CompatiblityExpander>(context);
}

}  // namespace stablehlo
}  // namespace mlir
