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
#include <stdbool.h>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iterator>
#include <optional>
#include <utility>

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/dialect/Version.h"
#include "stablehlo/transforms/PassUtils.h"  // IWYU pragma: keep
#include "stablehlo/transforms/Passes.h"

#define DEBUG_TYPE "stablehlo-compat"

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

bool fitsInIntegralType(int64_t size, IntegerType type) {
  if (type.isUnsigned()) {
    return llvm::isUIntN(type.getWidth(), size);
  } else {
    return llvm::isIntN(type.getWidth(), size);
  }
}

// If `type` is an integer type in which `size` doesn't fit, promote it to i32
// or i64 (depending on `size`).
Type promoteTypeForSize(Type type, int64_t size, OpBuilder& builder) {
  // Gather/Scatter should have an integer type, but we check just in case.
  auto intType = dyn_cast<IntegerType>(type);
  if (!intType || fitsInIntegralType(size, intType)) {
    return type;
  }
  if (fitsInIntegralType(size, builder.getI32Type())) {
    return builder.getI32Type();
  }
  return builder.getI64Type();
}

// If `indices_batching_dims` and `updated_index_map` are both sorted, then the
// `indices_are_sorted` property is preserved.
//
// This is because each concatenated iota is monotonically increasing, sorted
// indices batching dims mean their order corresponds to the order of batching
// dims in the operand, and a sorted updated start index map means the order of
// the index vector dim corresponds to the order of operand dims.
bool getUpdatedIndicesAreSorted(bool indices_are_sorted,
                                ArrayRef<int64_t> indices_batching_dims,
                                ArrayRef<int64_t> updated_index_map) {
  return indices_are_sorted && llvm::is_sorted(indices_batching_dims) &&
         llvm::is_sorted(updated_index_map);
}

// Returns an updated indices tensor such that an `IotaOp` is prepended for each
// dim in `indicesBatchingDims` with a `ConcatenateOp`.
//
// If `indexVectorDim` is equal to the rank of `indices`, it is reshaped to have
// a trailing dimension of size 1 so it can be concatenated with the `IotaOp`s.
Value createConcatIndices(Value indices, int64_t indexVectorDim,
                          ArrayRef<int64_t> indicesBatchingDims,
                          PatternRewriter& rewriter) {
  Location loc = indices.getLoc();
  auto indicesType = cast<RankedTensorType>(indices.getType());
  Type elementType = indicesType.getElementType();

  // The batching dim sizes might not fit in the existing element type,
  // in which case we need to promote it.
  for (int64_t batchingDim : indicesBatchingDims) {
    elementType = promoteTypeForSize(
        elementType, indicesType.getDimSize(batchingDim), rewriter);
  }
  if (elementType != indicesType.getElementType()) {
    indicesType = RankedTensorType::get(indicesType.getShape(), elementType);
    indices = ConvertOp::create(rewriter, loc, indicesType, indices);
  }

  bool indexVectorDimOnLastDim = indexVectorDim == indicesType.getRank();
  SmallVector<int64_t> iotaShape(indicesType.getShape());
  if (indexVectorDimOnLastDim) {
    iotaShape.push_back(1);
  } else {
    iotaShape[indexVectorDim] = 1;
  }
  auto iotaType = RankedTensorType::get(iotaShape, elementType);

  if (indexVectorDimOnLastDim) {
    indices = ReshapeOp::create(rewriter, loc, iotaType, indices);
  }

  SmallVector<Value> indicesToConcat;
  indicesToConcat.reserve(indicesBatchingDims.size() + 1);
  for (int64_t batchingDim : indicesBatchingDims) {
    indicesToConcat.push_back(
        IotaOp::create(rewriter, loc, iotaType, batchingDim));
  }
  indicesToConcat.push_back(indices);
  return ConcatenateOp::create(rewriter, loc, indicesToConcat, indexVectorDim);
}

//===----------------------------------------------------------------------===//
// Patterns (non DRR)
//===----------------------------------------------------------------------===//

// Converts a `GatherOp` with batching dims to a `GatherOp` without batching
// dims, such that each batching dim becomes a collapsed slice dim with a
// corresponding `IotaOp` concatenated to the start indices.
struct GatherWithBatchingDimsExpander : public OpRewritePattern<GatherOp> {
  using OpRewritePattern<GatherOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GatherOp op,
                                PatternRewriter& rewriter) const override {
    GatherDimensionNumbersAttr dimNumbers = op.getDimensionNumbers();
    ArrayRef<int64_t> operandBatchingDims = dimNumbers.getOperandBatchingDims();
    ArrayRef<int64_t> startIndicesBatchingDims =
        dimNumbers.getStartIndicesBatchingDims();
    if (operandBatchingDims.empty()) {
      return rewriter.notifyMatchFailure(op, [](Diagnostic& diag) {
        diag << "gather op has no batching dims";
      });
    }

    if (!op.getStartIndices().getType().hasStaticShape()) {
      return rewriter.notifyMatchFailure(op, [](Diagnostic& diag) {
        diag << "gather op has start indices with dynamic shape, can't expand";
      });
    }

    SmallVector<int64_t> newCollapsedSliceDims = mergeSortedDims(
        operandBatchingDims, dimNumbers.getCollapsedSliceDims());
    SmallVector<int64_t> newStartIndexMap =
        llvm::to_vector(llvm::concat<const int64_t>(
            operandBatchingDims, dimNumbers.getStartIndexMap()));
    Value newIndices = createConcatIndices(op.getStartIndices(),
                                           dimNumbers.getIndexVectorDim(),
                                           startIndicesBatchingDims, rewriter);
    rewriter.replaceOpWithNewOp<GatherOp>(
        op, op.getOperand(), newIndices,
        GatherDimensionNumbersAttr::get(
            op.getContext(), dimNumbers.getOffsetDims(), newCollapsedSliceDims,
            /*operandBatchingDims=*/{}, /*startIndicesBatchingDims=*/{},
            newStartIndexMap, dimNumbers.getIndexVectorDim()),
        op.getSliceSizes(),
        getUpdatedIndicesAreSorted(op.getIndicesAreSorted(),
                                   startIndicesBatchingDims, newStartIndexMap));

    return success();
  }
};

// Converts a `ScatterOp` with batching dims to a `ScatterOp` without batching
// dims, such that each batching dim becomes an inserted window dim with a
// corresponding `IotaOp` concatenated to the scatter indices.
struct ScatterWithBatchingDimsExpander : public OpRewritePattern<ScatterOp> {
  using OpRewritePattern<ScatterOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ScatterOp op,
                                PatternRewriter& rewriter) const override {
    ScatterDimensionNumbersAttr dimNumbers = op.getScatterDimensionNumbers();
    ArrayRef<int64_t> inputBatchingDims = dimNumbers.getInputBatchingDims();
    ArrayRef<int64_t> scatterIndicesBatchingDims =
        dimNumbers.getScatterIndicesBatchingDims();
    if (inputBatchingDims.empty()) {
      return rewriter.notifyMatchFailure(op, [](Diagnostic& diag) {
        diag << "scatter op has no batching dims";
      });
    }

    if (!op.getScatterIndices().getType().hasStaticShape()) {
      return rewriter.notifyMatchFailure(op, [](Diagnostic& diag) {
        diag << "gather op has start indices with dynamic shape, can't expand";
      });
    }

    SmallVector<int64_t> newInsertedWindowDims =
        mergeSortedDims(inputBatchingDims, dimNumbers.getInsertedWindowDims());
    SmallVector<int64_t> newScatterDimsToOperandDims =
        llvm::to_vector(llvm::concat<const int64_t>(
            inputBatchingDims, dimNumbers.getScatterDimsToOperandDims()));
    Value newIndices = createConcatIndices(
        op.getScatterIndices(), dimNumbers.getIndexVectorDim(),
        scatterIndicesBatchingDims, rewriter);
    auto newScatterOp = ScatterOp::create(
        rewriter, op.getLoc(), op->getResultTypes(), op.getInputs(), newIndices,
        op.getUpdates(),
        ScatterDimensionNumbersAttr::get(
            op.getContext(), dimNumbers.getUpdateWindowDims(),
            newInsertedWindowDims,
            /*inputBatchingDims=*/{}, /*scatterIndicesBatchingDims=*/{},
            newScatterDimsToOperandDims, dimNumbers.getIndexVectorDim()),
        getUpdatedIndicesAreSorted(op.getIndicesAreSorted(),
                                   scatterIndicesBatchingDims,
                                   newScatterDimsToOperandDims),
        op.getUniqueIndices());

    newScatterOp.getUpdateComputation().takeBody(op.getUpdateComputation());
    rewriter.replaceOp(op, newScatterOp.getResults());

    return success();
  }
};

// FileLineColRange locations are a forward incompatibility in upstream MLIR.
// This pattern removes the precise start/end range information and converts
// all FileLineColRange locations to forward compatible FileLineColLoc
// locations.
struct FileLineColRangeToLoc : public OpRewritePattern<ModuleOp> {
  using OpRewritePattern<ModuleOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ModuleOp op,
                                PatternRewriter& rewriter) const override {
    bool changed = false;
    mlir::AttrTypeReplacer replacer;
    replacer.addReplacement(
        [&](FileLineColLoc flcLoc) -> std::optional<Location> {
          // Skip if it's actually a FileLineColLoc
          if (isStrictFileLineColLoc(flcLoc)) return flcLoc;

          // Replace FileLineColRange with FileLineColLoc
          changed = true;
          auto newFlcLoc =
              FileLineColLoc::get(flcLoc.getFilename(), flcLoc.getStartLine(),
                                  flcLoc.getStartColumn());
          LLVM_DEBUG(llvm::dbgs() << "Rewriting FLC " << flcLoc << " -> "
                                  << newFlcLoc << "\n");
          return newFlcLoc;
        });

    // Call this on the module to update all locations in the module.
    // This should be safe since this pass is declared as a ModuleOp level pass
    // in the pass TD file, so no async issues.
    replacer.recursivelyReplaceElementsIn(op,
                                          /*replaceAttrs=*/false,
                                          /*replaceLocs=*/true,
                                          /*replaceTypes=*/false);

    return success(changed);
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
      const StablehloCompatibilityExpanderPassOptions& opts)
      : StablehloCompatibilityExpanderPassBase<
            StablehloCompatibilityExpanderPass>(opts) {}

 public:
  LogicalResult initialize(MLIRContext* context) override {
    auto targetVersion = validateTargetVersion(targetVersionOption);

    config.setUseTopDownTraversal(true);

    RewritePatternSet patterns_(context);
    populateStablehloCompatibilityExpanderPatterns(context, &patterns_,
                                                   targetVersion);
    patterns = std::move(patterns_);
    return success();
  }

  void runOnOperation() override {
    auto module = getOperation();

    // Apply to both the module and its children
    if (failed(
            applyOpPatternsGreedily(module.getOperation(), patterns, config)) ||
        failed(applyPatternsGreedily(module, patterns, config))) {
      module.emitError(
          "Failed to converge StableHLOCompatibilityExpanderPass in ")
          << config.getMaxIterations() << " iterations";
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
    MLIRContext* context, RewritePatternSet* patterns,
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

  // MLIR Upstream FileLineColRange introduced ~v1.8.4
  // Conservatively use 1.9.0 since StableHLO passes require major versions for
  // incompats.
  if (targetVersion < vhlo::Version(1, 9, 0))
    patterns->add<FileLineColRangeToLoc>(context);
}

}  // namespace stablehlo
}  // namespace mlir
