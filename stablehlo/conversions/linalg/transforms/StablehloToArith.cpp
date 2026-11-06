/* Copyright 2022 The IREE Authors
   Copyright 2023 OpenXLA Authors. All Rights Reserved.

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

#include <memory>
#include <utility>

#include "llvm/ADT/STLFunctionalExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo/conversions/linalg/transforms/MapStablehloToScalarOp.h"
#include "stablehlo/conversions/linalg/transforms/Rewriters.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::stablehlo {
namespace {

struct ScalarHloToFuncPattern final : ConversionPattern {
  ScalarHloToFuncPattern(TypeConverter& converter, MLIRContext* context,
                         PatternBenefit benefit = 1)
      : ConversionPattern(converter, ReturnOp::getOperationName(), benefit,
                          context) {}
  LogicalResult matchAndRewrite(
      Operation* op, ArrayRef<Value> /*operands*/,
      ConversionPatternRewriter& rewriter) const override {
    if (!isa<mlir::func::FuncOp>(op->getParentOp())) {
      return rewriter.notifyMatchFailure(op,
                                         "Return must be inside a function");
    }
    rewriter.replaceOpWithNewOp<mlir::func::ReturnOp>(op, op->getOperands());
    return success();
  }
};

struct ScalarHloToArithmeticPattern final : ConversionPattern {
  ScalarHloToArithmeticPattern(
      TypeConverter& converter, StringRef rootName, MLIRContext* context,
      llvm::function_ref<bool(Operation*)> filterFn = nullptr,
      PatternBenefit benefit = 1)
      : ConversionPattern(converter, rootName, benefit, context),
        filterFn(filterFn) {}

  LogicalResult matchAndRewrite(
      Operation* op, ArrayRef<Value> operands,
      ConversionPatternRewriter& rewriter) const override {
    if (filterFn && !filterFn(op)) return failure();

    auto isScalar = [](Value v) {
      return cast<ShapedType>(v.getType()).getRank() == 0;
    };

    if (!llvm::all_of(operands, isScalar))
      return rewriter.notifyMatchFailure(op, "All operands must be scalar.");

    Location loc = op->getLoc();

    auto resultTy = dyn_cast_or_null<ShapedType>(
        getTypeConverter()->convertType(op->getResultTypes().front()));
    if (!resultTy) return failure();

    SmallVector<Value> scalarOperands;
    scalarOperands.reserve(operands.size());
    for (Value operand : operands) {
      scalarOperands.push_back(
          tensor::ExtractOp::create(rewriter, loc, operand, ValueRange()));
    }
    Value scalarResult = StablehloOpToStdScalarOp::mapOp(
        op, resultTy.getElementType(), scalarOperands, &rewriter);
    if (!scalarResult) return failure();
    rewriter.replaceOpWithNewOp<tensor::FromElementsOp>(op, resultTy,
                                                        scalarResult);
    return success();
  }

 private:
  llvm::function_ref<bool(Operation*)> filterFn;
};

void addPattern(RewritePatternSet* patterns,
                std::unique_ptr<RewritePattern> pattern) {
  patterns->add(std::move(pattern));
}

template <typename... OpTypes>
void populateScalarArithPatterns(
    MLIRContext* context, TypeConverter& typeConverter,
    RewritePatternSet* patterns,
    llvm::function_ref<bool(Operation*)> filterFn) {
  (addPattern(patterns, std::make_unique<ScalarHloToArithmeticPattern>(
                            typeConverter, OpTypes::getOperationName(), context,
                            filterFn)),
   ...);
}

}  // namespace

namespace detail {
void populateScalarHloToArithConversionPatterns(
    MLIRContext* context, TypeConverter& typeConverter,
    RewritePatternSet* patterns,
    llvm::function_ref<bool(Operation*)> filterFn) {
  // TODO(#12678): Handle the XLA rng op.
  populateScalarArithPatterns<
      AbsOp, AddOp, AndOp, Atan2Op, BitcastConvertOp, CbrtOp, CeilOp, ClampOp,
      ClzOp, CompareOp, ComplexOp, ConvertOp, CosineOp, DivOp, ExpOp, Expm1Op,
      Exp2Op, FloorOp, FmaOp, ImagOp, IsFiniteOp, Log1pOp, Log2Op, LogOp,
      LogisticOp, MaxOp, MinOp, MulOp, NegOp, NotOp, OrOp, PopulationCountOp,
      PowOp, RealOp, ReducePrecisionOp, RemOp, RoundNearestEvenOp, RoundOp,
      RsqrtOp, SelectOp, ShiftLeftOp, ShiftRightArithmeticOp,
      ShiftRightLogicalOp, SignOp, SineOp, SqrtOp, SubtractOp, TanOp, TanhOp,
      XorOp>(context, typeConverter, patterns, filterFn);
  addPattern(patterns,
             std::make_unique<ScalarHloToFuncPattern>(typeConverter, context));
}
}  // namespace detail

}  // namespace mlir::stablehlo
