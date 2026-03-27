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

#include <utility>

#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/transforms/PassUtils.h"
#include "stablehlo/transforms/Passes.h"

namespace mlir {
namespace stablehlo {
#define GEN_PASS_DEF_STABLEHLOCOMPLEXMATHEXPANDERPASS
#include "stablehlo/transforms/Passes.h.inc"

namespace {

static Value getConstantLikeMaxFiniteValue(OpBuilder &b, Location loc,
                                           Value val) {
  auto ty = cast<FloatType>(getElementTypeOrSelf(val.getType()));
  return getConstantLike(
      b, loc, llvm::APFloat::getLargest(ty.getFloatSemantics()), val);
}

static Value getConstantLikeInfValue(OpBuilder &b, Location loc, Value val,
                                     bool negative) {
  auto ty = cast<FloatType>(getElementTypeOrSelf(val.getType()));
  return getConstantLike(
      b, loc, llvm::APFloat::getInf(ty.getFloatSemantics(), negative), val);
}

static bool hasComplexElementType(Value val) {
  auto shapedTy = dyn_cast<ShapedType>(val.getType());
  return shapedTy && isa<ComplexType>(shapedTy.getElementType());
}

//===----------------------------------------------------------------------===//
// Complex arithmetic decomposition patterns (mode=all)
//===----------------------------------------------------------------------===//

// (a+bi) + (c+di) = (a+c) + (b+d)i
struct ExpandComplexAdd : public OpRewritePattern<AddOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AddOp op,
                                PatternRewriter &rewriter) const override {
    if (!hasComplexElementType(op.getLhs())) return failure();

    Location loc = op.getLoc();
    Value lhsReal = rewriter.create<RealOp>(loc, op.getLhs());
    Value lhsImag = rewriter.create<ImagOp>(loc, op.getLhs());
    Value rhsReal = rewriter.create<RealOp>(loc, op.getRhs());
    Value rhsImag = rewriter.create<ImagOp>(loc, op.getRhs());

    Value resultReal = rewriter.create<AddOp>(loc, lhsReal, rhsReal);
    Value resultImag = rewriter.create<AddOp>(loc, lhsImag, rhsImag);

    rewriter.replaceOpWithNewOp<ComplexOp>(op, resultReal, resultImag);
    return success();
  }
};

// (a+bi) - (c+di) = (a-c) + (b-d)i
struct ExpandComplexSubtract : public OpRewritePattern<SubtractOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(SubtractOp op,
                                PatternRewriter &rewriter) const override {
    if (!hasComplexElementType(op.getLhs())) return failure();

    Location loc = op.getLoc();
    Value lhsReal = rewriter.create<RealOp>(loc, op.getLhs());
    Value lhsImag = rewriter.create<ImagOp>(loc, op.getLhs());
    Value rhsReal = rewriter.create<RealOp>(loc, op.getRhs());
    Value rhsImag = rewriter.create<ImagOp>(loc, op.getRhs());

    Value resultReal = rewriter.create<SubtractOp>(loc, lhsReal, rhsReal);
    Value resultImag = rewriter.create<SubtractOp>(loc, lhsImag, rhsImag);

    rewriter.replaceOpWithNewOp<ComplexOp>(op, resultReal, resultImag);
    return success();
  }
};

// (a+bi) * (c+di) = (ac - bd) + (ad + bc)i
struct ExpandComplexMultiply : public OpRewritePattern<MulOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MulOp op,
                                PatternRewriter &rewriter) const override {
    if (!hasComplexElementType(op.getLhs())) return failure();

    Location loc = op.getLoc();
    Value a = rewriter.create<RealOp>(loc, op.getLhs());
    Value b = rewriter.create<ImagOp>(loc, op.getLhs());
    Value c = rewriter.create<RealOp>(loc, op.getRhs());
    Value d = rewriter.create<ImagOp>(loc, op.getRhs());

    Value ac = rewriter.create<MulOp>(loc, a, c);
    Value bd = rewriter.create<MulOp>(loc, b, d);
    Value ad = rewriter.create<MulOp>(loc, a, d);
    Value bc = rewriter.create<MulOp>(loc, b, c);

    Value resultReal = rewriter.create<SubtractOp>(loc, ac, bd);
    Value resultImag = rewriter.create<AddOp>(loc, ad, bc);

    rewriter.replaceOpWithNewOp<ComplexOp>(op, resultReal, resultImag);
    return success();
  }
};

// (a+bi) / (c+di)
// Uses Smith's method for numerical stability:
//   if |d| <= |c|: r = d/c, denom = c + d*r
//     result = ((a + b*r) / denom) + ((b - a*r) / denom)i
//   else: r = c/d, denom = d + c*r
//     result = ((a*r + b) / denom) + ((b*r - a) / denom)i
struct ExpandComplexDivide : public OpRewritePattern<DivOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(DivOp op,
                                PatternRewriter &rewriter) const override {
    if (!hasComplexElementType(op.getLhs())) return failure();

    Location loc = op.getLoc();
    Value a = rewriter.create<RealOp>(loc, op.getLhs());
    Value b = rewriter.create<ImagOp>(loc, op.getLhs());
    Value c = rewriter.create<RealOp>(loc, op.getRhs());
    Value d = rewriter.create<ImagOp>(loc, op.getRhs());

    Value absC = rewriter.create<AbsOp>(loc, c);
    Value absD = rewriter.create<AbsOp>(loc, d);

    // |d| <= |c| branch: r = d/c, denom = c + d*r
    Value rCd = rewriter.create<DivOp>(loc, d, c);
    Value denomCd = rewriter.create<AddOp>(
        loc, c, rewriter.create<MulOp>(loc, d, rCd));
    Value realCd = rewriter.create<DivOp>(
        loc,
        rewriter.create<AddOp>(loc, a, rewriter.create<MulOp>(loc, b, rCd)),
        denomCd);
    Value imagCd = rewriter.create<DivOp>(
        loc,
        rewriter.create<SubtractOp>(loc, b,
                                    rewriter.create<MulOp>(loc, a, rCd)),
        denomCd);

    // |d| > |c| branch: r = c/d, denom = d + c*r
    Value rDc = rewriter.create<DivOp>(loc, c, d);
    Value denomDc = rewriter.create<AddOp>(
        loc, d, rewriter.create<MulOp>(loc, c, rDc));
    Value realDc = rewriter.create<DivOp>(
        loc,
        rewriter.create<AddOp>(loc, rewriter.create<MulOp>(loc, a, rDc), b),
        denomDc);
    Value imagDc = rewriter.create<DivOp>(
        loc,
        rewriter.create<SubtractOp>(loc, rewriter.create<MulOp>(loc, b, rDc),
                                    a),
        denomDc);

    // Select based on |d| <= |c|
    Value cond = rewriter.create<CompareOp>(
        loc, absD, absC, ComparisonDirection::LE);
    Value resultReal = rewriter.create<SelectOp>(loc, cond, realCd, realDc);
    Value resultImag = rewriter.create<SelectOp>(loc, cond, imagCd, imagDc);

    rewriter.replaceOpWithNewOp<ComplexOp>(op, resultReal, resultImag);
    return success();
  }
};

// -(a+bi) = (-a) + (-b)i
struct ExpandComplexNegate : public OpRewritePattern<NegOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(NegOp op,
                                PatternRewriter &rewriter) const override {
    if (!hasComplexElementType(op.getOperand())) return failure();

    Location loc = op.getLoc();
    Value real = rewriter.create<RealOp>(loc, op.getOperand());
    Value imag = rewriter.create<ImagOp>(loc, op.getOperand());

    Value negReal = rewriter.create<NegOp>(loc, real);
    Value negImag = rewriter.create<NegOp>(loc, imag);

    rewriter.replaceOpWithNewOp<ComplexOp>(op, negReal, negImag);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct StablehloComplexMathExpanderPass
    : public impl::StablehloComplexMathExpanderPassBase<
          StablehloComplexMathExpanderPass> {
  using StablehloComplexMathExpanderPassBase::
      StablehloComplexMathExpanderPassBase;

 public:
  LogicalResult initialize(MLIRContext *context) override {
    config.setUseTopDownTraversal(true);
    RewritePatternSet patterns_(context);
    populateStablehloComplexMathExpanderPatterns(context, &patterns_);
    if (mode == "all") {
      patterns_.add<ExpandComplexAdd, ExpandComplexSubtract,
                    ExpandComplexMultiply, ExpandComplexDivide,
                    ExpandComplexNegate>(context);
    }
    patterns = std::move(patterns_);
    return success();
  }

  void runOnOperation() override {
    auto func = getOperation();
    if (failed(applyPatternsGreedily(func, patterns, config))) {
      func.emitError("Failed to converge StableHLOComplexMathExpanderPass in ")
          << config.getMaxIterations() << " iterations";
      signalPassFailure();
    }
  }

 private:
  FrozenRewritePatternSet patterns;
  GreedyRewriteConfig config;
};

#include "stablehlo/transforms/StablehloComplexMathExpanderPatterns.h.inc"

}  // namespace

void populateStablehloComplexMathExpanderPatterns(MLIRContext *context,
                                                  RewritePatternSet *patterns) {
  populateWithGenerated(*patterns);
}

void populateStablehloComplexArithmeticExpanderPatterns(
    MLIRContext *context, RewritePatternSet *patterns) {
  patterns->add<ExpandComplexAdd, ExpandComplexSubtract,
                ExpandComplexMultiply, ExpandComplexDivide,
                ExpandComplexNegate>(context);
}

}  // namespace stablehlo
}  // namespace mlir
