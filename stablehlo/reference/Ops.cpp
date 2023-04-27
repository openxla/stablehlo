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

#include "stablehlo/reference/Ops.h"

#include <algorithm>
#include <numeric>

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/DebugStringHelper.h"
#include "stablehlo/reference/Element.h"
#include "stablehlo/reference/Errors.h"
#include "stablehlo/reference/Types.h"

namespace mlir {
namespace stablehlo {
namespace {

Index evalIndices(ArrayRef<Tensor> runtimeIndices) {
  Index index(runtimeIndices.size());
  for (size_t i = 0; i < runtimeIndices.size(); ++i)
    index[i] = runtimeIndices[i].get({}).getIntegerValue().getSExtValue();
  return index;
}

}  // namespace

SmallVector<Tensor> eval(
    Region &region, ArrayRef<Tensor> args, Scope *parent,
    llvm::function_ref<llvm::Error(Operation &, Scope &)> fallback) {
  Block &block = region.front();
  if (block.getArguments().size() != args.size())
    report_fatal_error(invalidArgument(
        "Expected same number of block arguments and runtime arguments (%d)",
        args.size()));

  Scope scope(parent);
  scope.add(block.getArguments(), args);

  for (Operation &op : block) {
    if (auto absOp = dyn_cast<AbsOp>(op)) {
      Tensor runtimeOperand = scope.find(absOp.getOperand());
      Tensor runtimeResult = evalAbsOp(runtimeOperand, absOp.getType());
      scope.add(op.getResults(), {runtimeResult});
    } else if (auto addOp = dyn_cast<AddOp>(op)) {
      Tensor runtimeLhs = scope.find(addOp.getLhs());
      Tensor runtimeRhs = scope.find(addOp.getRhs());
      Tensor runtimeResult = evalAddOp(runtimeLhs, runtimeRhs, addOp.getType());
      scope.add(op.getResults(), {runtimeResult});
    } else if (auto andOp = dyn_cast<AndOp>(op)) {
      Tensor runtimeLhs = scope.find(andOp.getLhs());
      Tensor runtimeRhs = scope.find(andOp.getRhs());
      Tensor runtimeResult = evalAndOp(runtimeLhs, runtimeRhs, andOp.getType());
      scope.add(op.getResults(), {runtimeResult});
    } else if (auto atan2Op = dyn_cast<Atan2Op>(op)) {
      Tensor runtimeLhs = scope.find(atan2Op.getLhs());
      Tensor runtimeRhs = scope.find(atan2Op.getRhs());
      Tensor runtimeResult =
          evalAtan2Op(runtimeLhs, runtimeRhs, atan2Op.getType());
      scope.add(op.getResults(), {runtimeResult});
    } else if (auto broadcastInDimOp = dyn_cast<BroadcastInDimOp>(op)) {
      Tensor runtimeOperand = scope.find(broadcastInDimOp.getOperand());
      auto broadcastDimensions =
          Axes(broadcastInDimOp.getBroadcastDimensions());
      Tensor runtimeResult = evalBroadcastInDimOp(
          runtimeOperand, broadcastDimensions, broadcastInDimOp.getType());
      scope.add(op.getResults(), {runtimeResult});
    } else if (auto caseOp = dyn_cast<CaseOp>(op)) {
      Tensor runtimeIndex = scope.find(caseOp.getIndex());
      auto runtimeResults =
          evalCaseOp(runtimeIndex, caseOp.getBranches(), scope);
      scope.add(op.getResults(), {runtimeResults});
    } else if (auto ceilOp = dyn_cast<CeilOp>(op)) {
      Tensor runtimeOperand = scope.find(ceilOp.getOperand());
      Tensor runtimeResult = evalCeilOp(runtimeOperand, ceilOp.getType());
      scope.add(op.getResults(), {runtimeResult});
    } else if (auto clampOp = dyn_cast<ClampOp>(op)) {
      Tensor runtimeMin = scope.find(clampOp.getMin());
      Tensor runtimeOperand = scope.find(clampOp.getOperand());
      Tensor runtimeMax = scope.find(clampOp.getMax());
      Tensor runtimeResult = evalClampOp(runtimeMin, runtimeOperand, runtimeMax,
                                         clampOp.getType());
      scope.add(op.getResults(), {runtimeResult});
    } else if (auto clzOp = dyn_cast<ClzOp>(op)) {
      Tensor runtimeOperand = scope.find(clzOp.getOperand());
      Tensor runtimeResult = evalClzOp(runtimeOperand, clzOp.getType());
      scope.add(op.getResults(), {runtimeResult});
    } else if (auto compareOp = dyn_cast<CompareOp>(op)) {
      Tensor runtimeLhs = scope.find(compareOp.getLhs());
      Tensor runtimeRhs = scope.find(compareOp.getRhs());
      auto comparisonDirection = compareOp.getComparisonDirection();
      auto runtimeResult = evalCompareOp(
          runtimeLhs, runtimeRhs, comparisonDirection, compareOp.getType());
      scope.add(op.getResults(), {runtimeResult});
    } else if (auto concatenateOp = dyn_cast<ConcatenateOp>(op)) {
      auto runtimeOperands = scope.find(concatenateOp.getOperands());
      Tensor runtimeResult =
          evalConcatenateOp(runtimeOperands, concatenateOp.getDimension(),
                            concatenateOp.getType());
      scope.add(op.getResults(), {runtimeResult});
    } else if (auto constantOp = dyn_cast<ConstantOp>(op)) {
      Tensor runtimeResult = evalConstantOp(constantOp.getValue());
      scope.add(op.getResults(), {runtimeResult});
    } else if (auto convertOp = dyn_cast<ConvertOp>(op)) {
      Tensor runtimeOperand = scope.find(convertOp.getOperand());
      Tensor runtimeResult = evalConvertOp(runtimeOperand, convertOp.getType());
      scope.add(op.getResults(), {runtimeResult});
    } else if (auto cosineOp = dyn_cast<CosineOp>(op)) {
      Tensor runtimeOperand = scope.find(cosineOp.getOperand());
      Tensor runtimeResult = evalCosineOp(runtimeOperand, cosineOp.getType());
      scope.add(op.getResults(), {runtimeResult});
    } else if (auto divideOp = dyn_cast<DivOp>(op)) {
      Tensor runtimeLhs = scope.find(divideOp.getLhs());
      Tensor runtimeRhs = scope.find(divideOp.getRhs());
      Tensor runtimeResult =
          evalDivideOp(runtimeLhs, runtimeRhs, divideOp.getType());
      scope.add(op.getResults(), {runtimeResult});
    } else if (auto dynamicSliceOp = dyn_cast<DynamicSliceOp>(op)) {
      Tensor runtimeOperand = scope.find(dynamicSliceOp.getOperand());
      SmallVector<Tensor> runtimeStartIndices =
          scope.find(dynamicSliceOp.getStartIndices());
      auto runtimeSliceSizes = Sizes(dynamicSliceOp.getSliceSizes());
      Tensor runtimeResult =
          evalDynamicSliceOp(runtimeOperand, runtimeStartIndices,
                             runtimeSliceSizes, dynamicSliceOp.getType());
      scope.add(op.getResults(), {runtimeResult});
    } else if (auto dynamicUpdateSliceOp = dyn_cast<DynamicUpdateSliceOp>(op)) {
      Tensor runtimeOperand = scope.find(dynamicUpdateSliceOp.getOperand());
      Tensor runtimeUpdate = scope.find(dynamicUpdateSliceOp.getUpdate());
      SmallVector<Tensor> runtimeStartIndices =
          scope.find(dynamicUpdateSliceOp.getStartIndices());
      Tensor runtimeResult = evalDynamicUpdateSliceOp(
          runtimeOperand, runtimeUpdate, runtimeStartIndices,
          dynamicUpdateSliceOp.getType());
      scope.add(op.getResults(), {runtimeResult});
    } else if (auto expOp = dyn_cast<ExpOp>(op)) {
      Tensor runtimeOperand = scope.find(expOp.getOperand());
      Tensor runtimeResult = evalExponentialOp(runtimeOperand, expOp.getType());
      scope.add(op.getResults(), {runtimeResult});
    } else if (auto floorOp = dyn_cast<FloorOp>(op)) {
      Tensor runtimeOperand = scope.find(floorOp.getOperand());
      Tensor runtimeResult = evalFloorOp(runtimeOperand, floorOp.getType());
      scope.add(op.getResults(), {runtimeResult});
    } else if (auto getDimensionSizeOp = dyn_cast<GetDimensionSizeOp>(op)) {
      Tensor runtimeOperand = scope.find(getDimensionSizeOp.getOperand());
      auto runtimeResults = evalGetDimensionSizeOp(
          runtimeOperand, getDimensionSizeOp.getDimension(),
          getDimensionSizeOp.getType());
      scope.add(op.getResults(), runtimeResults);
    } else if (auto ifOp = dyn_cast<IfOp>(op)) {
      Tensor runtimePred = scope.find(ifOp.getPred());
      auto runtimeResults = evalIfOp(runtimePred, ifOp.getTrueBranch(),
                                     ifOp.getFalseBranch(), scope);
      scope.add(op.getResults(), runtimeResults);
    } else if (auto imagOp = dyn_cast<ImagOp>(op)) {
      Tensor runtimeOperand = scope.find(imagOp.getOperand());
      Tensor runtimeResult = evalImagOp(runtimeOperand, imagOp.getType());
      scope.add(op.getResults(), {runtimeResult});
    } else if (auto iotaOp = dyn_cast<IotaOp>(op)) {
      Tensor runtimeResult =
          evalIotaOp(iotaOp.getIotaDimension(), iotaOp.getType());
      scope.add(op.getResults(), {runtimeResult});
    } else if (auto isFiniteOp = dyn_cast<IsFiniteOp>(op)) {
      Tensor runtimeOperand = scope.find(isFiniteOp.getOperand());
      Tensor runtimeResult =
          evalIsFiniteOp(runtimeOperand, isFiniteOp.getResult().getType());
      scope.add(op.getResults(), {runtimeResult});
    } else if (auto logOp = dyn_cast<LogOp>(op)) {
      Tensor runtimeOperand = scope.find(logOp.getOperand());
      Tensor runtimeResult = evalLogOp(runtimeOperand, logOp.getType());
      scope.add(op.getResults(), {runtimeResult});
    } else if (auto logisticOp = dyn_cast<LogisticOp>(op)) {
      Tensor runtimeOperand = scope.find(logisticOp.getOperand());
      Tensor runtimeResult =
          evalLogisticOp(runtimeOperand, logisticOp.getType());
      scope.add(op.getResults(), {runtimeResult});
    } else if (auto mapOp = dyn_cast<MapOp>(op)) {
      SmallVector<Tensor> runtimeInputs = scope.find(mapOp.getInputs());
      auto runtimeResults = evalMapOp(runtimeInputs, mapOp.getComputation(),
                                      scope, mapOp.getType());
      scope.add(op.getResults(), {runtimeResults});
    } else if (auto maxOp = dyn_cast<MaxOp>(op)) {
      Tensor runtimeLhs = scope.find(maxOp.getLhs());
      Tensor runtimeRhs = scope.find(maxOp.getRhs());
      Tensor runtimeResult = evalMaxOp(runtimeLhs, runtimeRhs, maxOp.getType());
      scope.add(op.getResults(), {runtimeResult});
    } else if (auto minOp = dyn_cast<MinOp>(op)) {
      Tensor runtimeLhs = scope.find(minOp.getLhs());
      Tensor runtimeRhs = scope.find(minOp.getRhs());
      Tensor runtimeResult = evalMinOp(runtimeLhs, runtimeRhs, minOp.getType());
      scope.add(op.getResults(), {runtimeResult});
    } else if (auto multiplyOp = dyn_cast<MulOp>(op)) {
      Tensor runtimeLhs = scope.find(multiplyOp.getLhs());
      Tensor runtimeRhs = scope.find(multiplyOp.getRhs());
      Tensor runtimeResult =
          evalMultiplyOp(runtimeLhs, runtimeRhs, multiplyOp.getType());
      scope.add(op.getResults(), {runtimeResult});
    } else if (auto negOp = dyn_cast<NegOp>(op)) {
      Tensor runtimeOperand = scope.find(negOp.getOperand());
      Tensor runtimeResult = evalNegOp(runtimeOperand, negOp.getType());
      scope.add(op.getResults(), {runtimeResult});
    } else if (auto notOp = dyn_cast<NotOp>(op)) {
      Tensor runtimeOperand = scope.find(notOp.getOperand());
      Tensor runtimeResult = evalNotOp(runtimeOperand, notOp.getType());
      scope.add(op.getResults(), {runtimeResult});
    } else if (auto orOp = dyn_cast<OrOp>(op)) {
      Tensor runtimeLhs = scope.find(orOp.getLhs());
      Tensor runtimeRhs = scope.find(orOp.getRhs());
      Tensor runtimeResult = evalOrOp(runtimeLhs, runtimeRhs, orOp.getType());
      scope.add(op.getResults(), {runtimeResult});
    } else if (auto padOp = dyn_cast<PadOp>(op)) {
      Tensor runtimeOperand = scope.find(padOp.getOperand());
      Tensor runtimePaddingValue = scope.find(padOp.getPaddingValue());
      auto edgePaddingLow = Sizes(padOp.getEdgePaddingLow());
      auto interiorPadding = Sizes(padOp.getInteriorPadding());
      Tensor runtimeResult =
          evalPadOp(runtimeOperand, runtimePaddingValue, edgePaddingLow,
                    interiorPadding, padOp.getType());
      scope.add(op.getResults(), {runtimeResult});
    } else if (auto populationCountOp = dyn_cast<PopulationCountOp>(op)) {
      Tensor runtimeOperand = scope.find(populationCountOp.getOperand());
      Tensor runtimeResult =
          evalPopulationCountOp(runtimeOperand, populationCountOp.getType());
      scope.add(op.getResults(), {runtimeResult});
    } else if (auto powerOp = dyn_cast<PowOp>(op)) {
      Tensor runtimeLhs = scope.find(powerOp.getLhs());
      Tensor runtimeRhs = scope.find(powerOp.getRhs());
      Tensor runtimeResult =
          evalPowerOp(runtimeLhs, runtimeRhs, powerOp.getType());
      scope.add(op.getResults(), {runtimeResult});
    } else if (auto realOp = dyn_cast<RealOp>(op)) {
      Tensor runtimeOperand = scope.find(realOp.getOperand());
      Tensor runtimeResult = evalRealOp(runtimeOperand, realOp.getType());
      scope.add(op.getResults(), {runtimeResult});
    } else if (auto remOp = dyn_cast<RemOp>(op)) {
      Tensor runtimeLhs = scope.find(remOp.getLhs());
      Tensor runtimeRhs = scope.find(remOp.getRhs());
      Tensor runtimeResult = evalRemOp(runtimeLhs, runtimeRhs, remOp.getType());
      scope.add(op.getResults(), {runtimeResult});
    } else if (auto reshapeOp = dyn_cast<ReshapeOp>(op)) {
      Tensor runtimeOperand = scope.find(reshapeOp.getOperand());
      Tensor runtimeResult = evalReshapeOp(runtimeOperand, reshapeOp.getType());
      scope.add(op.getResults(), {runtimeResult});
    } else if (auto returnOp = dyn_cast<func::ReturnOp>(op)) {
      return scope.find(returnOp.getOperands());
    } else if (auto returnOp = dyn_cast<ReturnOp>(op)) {
      return scope.find(returnOp.getResults());
    } else if (auto reverseOp = dyn_cast<ReverseOp>(op)) {
      Tensor runtimeOperand = scope.find(reverseOp.getOperand());
      auto dimensions = Axes(reverseOp.getDimensions());
      Tensor runtimeResult =
          evalReverseOp(runtimeOperand, dimensions, reverseOp.getType());
      scope.add(op.getResults(), {runtimeResult});
    } else if (auto roundOp = dyn_cast<RoundOp>(op)) {
      Tensor runtimeOperand = scope.find(roundOp.getOperand());
      Tensor runtimeResult = evalRoundOp(runtimeOperand, roundOp.getType());
      scope.add(op.getResults(), {runtimeResult});
    } else if (auto roundNearestEvenOp = dyn_cast<RoundNearestEvenOp>(op)) {
      Tensor runtimeOperand = scope.find(roundNearestEvenOp.getOperand());
      Tensor runtimeResult =
          evalRoundNearestEvenOp(runtimeOperand, roundNearestEvenOp.getType());
      scope.add(op.getResults(), {runtimeResult});
    } else if (auto rsqrtOp = dyn_cast<RsqrtOp>(op)) {
      Tensor runtimeOperand = scope.find(rsqrtOp.getOperand());
      Tensor runtimeResult = evalRsqrtOp(runtimeOperand, rsqrtOp.getType());
      scope.add(op.getResults(), {runtimeResult});
    } else if (auto selectOp = dyn_cast<SelectOp>(op)) {
      Tensor runtimeResult = evalSelectOp(
          scope.find(selectOp.getPred()), scope.find(selectOp.getOnTrue()),
          scope.find(selectOp.getOnFalse()), selectOp.getType());
      scope.add(op.getResults(), {runtimeResult});
    } else if (auto shiftLeftOp = dyn_cast<ShiftLeftOp>(op)) {
      Tensor runtimeLhs = scope.find(shiftLeftOp.getLhs());
      Tensor runtimeRhs = scope.find(shiftLeftOp.getRhs());
      Tensor runtimeResult =
          evalShiftLeftOp(runtimeLhs, runtimeRhs, shiftLeftOp.getType());
      scope.add(op.getResults(), {runtimeResult});
    } else if (auto shiftRightArithmeticOp =
                   dyn_cast<ShiftRightArithmeticOp>(op)) {
      Tensor runtimeLhs = scope.find(shiftRightArithmeticOp.getLhs());
      Tensor runtimeRhs = scope.find(shiftRightArithmeticOp.getRhs());
      Tensor runtimeResult = evalShiftRightArithmeticOp(
          runtimeLhs, runtimeRhs, shiftRightArithmeticOp.getType());
      scope.add(op.getResults(), {runtimeResult});
    } else if (auto shiftRightLogicalOp = dyn_cast<ShiftRightLogicalOp>(op)) {
      Tensor runtimeLhs = scope.find(shiftRightLogicalOp.getLhs());
      Tensor runtimeRhs = scope.find(shiftRightLogicalOp.getRhs());
      Tensor runtimeResult = evalShiftRightLogicalOp(
          runtimeLhs, runtimeRhs, shiftRightLogicalOp.getType());
      scope.add(op.getResults(), {runtimeResult});
    } else if (auto signOp = dyn_cast<SignOp>(op)) {
      Tensor runtimeOperand = scope.find(signOp.getOperand());
      Tensor runtimeResult = evalSignOp(runtimeOperand, signOp.getType());
      scope.add(op.getResults(), {runtimeResult});
    } else if (auto sineOp = dyn_cast<SineOp>(op)) {
      Tensor runtimeOperand = scope.find(sineOp.getOperand());
      Tensor runtimeResult = evalSineOp(runtimeOperand, sineOp.getType());
      scope.add(op.getResults(), {runtimeResult});
    } else if (auto sliceOp = dyn_cast<SliceOp>(op)) {
      Tensor runtimeOperand = scope.find(sliceOp.getOperand());
      auto startIndices = Sizes(sliceOp.getStartIndices());
      auto strides = Sizes(sliceOp.getStrides());
      Tensor runtimeResult =
          evalSliceOp(runtimeOperand, startIndices, strides, sliceOp.getType());
      scope.add(op.getResults(), {runtimeResult});
    } else if (auto sortOp = dyn_cast<SortOp>(op)) {
      SmallVector<Tensor> runtimeOperands = scope.find(sortOp.getInputs());
      auto runtimeResults =
          evalSortOp(runtimeOperands, sortOp.getDimension(),
                     sortOp.getIsStable(), sortOp.getComparator(), scope);
      scope.add(op.getResults(), {runtimeResults});
    } else if (auto sqrtOp = dyn_cast<SqrtOp>(op)) {
      Tensor runtimeOperand = scope.find(sqrtOp.getOperand());
      Tensor runtimeResult = evalSqrtOp(runtimeOperand, sqrtOp.getType());
      scope.add(op.getResults(), {runtimeResult});
    } else if (auto subtractOp = dyn_cast<SubtractOp>(op)) {
      Tensor runtimeLhs = scope.find(subtractOp.getLhs());
      Tensor runtimeRhs = scope.find(subtractOp.getRhs());
      Tensor runtimeResult =
          evalSubtractOp(runtimeLhs, runtimeRhs, subtractOp.getType());
      scope.add(op.getResults(), {runtimeResult});
    } else if (auto tanhOp = dyn_cast<TanhOp>(op)) {
      Tensor runtimeOperand = scope.find(tanhOp.getOperand());
      Tensor runtimeResult = evalTanhOp(runtimeOperand, tanhOp.getType());
      scope.add(op.getResults(), {runtimeResult});
    } else if (auto transposeOp = dyn_cast<TransposeOp>(op)) {
      Tensor runtimeOperand = scope.find(transposeOp.getOperand());
      auto permutation = Axes(transposeOp.getPermutation());
      Tensor runtimeResult =
          evalTransposeOp(runtimeOperand, permutation, transposeOp.getType());
      scope.add(op.getResults(), {runtimeResult});
    } else if (auto whileOp = dyn_cast<WhileOp>(op)) {
      SmallVector<Value> runtimeOperands(whileOp.getOperand().begin(),
                                         whileOp.getOperand().end());
      auto runtimeInputs = scope.find(runtimeOperands);
      auto runtimeResults = evalWhileOp(runtimeInputs, whileOp.getCond(),
                                        whileOp.getBody(), scope);
      scope.add(op.getResults(), runtimeResults);
    } else if (auto xorOp = dyn_cast<XorOp>(op)) {
      Tensor runtimeLhs = scope.find(xorOp.getLhs());
      Tensor runtimeRhs = scope.find(xorOp.getRhs());
      Tensor runtimeResult = evalXorOp(runtimeLhs, runtimeRhs, xorOp.getType());
      scope.add(op.getResults(), {runtimeResult});
    } else {
      if (!fallback)
        report_fatal_error(
            invalidArgument("Unsupported op: %s", debugString(op).c_str()));
      auto status = fallback(op, scope);
      if (status) llvm::report_fatal_error(std::move(status));
    }
  }

  llvm::report_fatal_error("Expected a terminator when evaluating a region");
}

Tensor evalAbsOp(const Tensor &operand, ShapedType resultType) {
  Tensor result(resultType);
  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it, abs(operand.get(*it)));
  return result;
}

Tensor evalAddOp(const Tensor &lhs, const Tensor &rhs, ShapedType resultType) {
  Tensor result(resultType);
  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it, lhs.get(*it) + rhs.get(*it));
  return result;
}

Tensor evalAndOp(const Tensor &lhs, const Tensor &rhs, ShapedType resultType) {
  Tensor result(resultType);
  for (auto it = lhs.index_begin(); it != lhs.index_end(); ++it)
    result.set(*it, lhs.get(*it) & rhs.get(*it));
  return result;
}

Tensor evalAtan2Op(const Tensor &lhs, const Tensor &rhs,
                   ShapedType resultType) {
  Tensor result(resultType);
  for (auto resultIt = result.index_begin(); resultIt != result.index_end();
       ++resultIt)
    result.set(*resultIt, atan2(lhs.get(*resultIt), rhs.get(*resultIt)));
  return result;
}

Tensor evalBroadcastInDimOp(const Tensor &operand, Axes broadcastDimensions,
                            ShapedType resultType) {
  Tensor result(resultType);
  auto operandShape = operand.getShape();
  for (auto resultIt = result.index_begin(); resultIt != result.index_end();
       ++resultIt) {
    Index operandIdx(operandShape.size());
    for (auto [operandDim, resultDim] : llvm::enumerate(broadcastDimensions))
      operandIdx[operandDim] =
          operandShape[operandDim] == 1 ? 0 : (*resultIt)[resultDim];
    result.set(*resultIt, operand.get(operandIdx));
  }
  return result;
}

SmallVector<Tensor> evalCaseOp(const Tensor &index, RegionRange branches,
                               Scope &scope) {
  int64_t idx = index.get({}).getIntegerValue().getSExtValue();
  if (idx < 0 || idx >= static_cast<int64_t>(branches.size()))
    idx = branches.size() - 1;
  return eval(*branches[idx], {}, &scope);
}

Tensor evalCeilOp(const Tensor &operand, ShapedType resultType) {
  Tensor result(resultType);
  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it, ceil(operand.get(*it)));
  return result;
}

Tensor evalClampOp(const Tensor &min, const Tensor &operand, const Tensor &max,
                   ShapedType resultType) {
  Tensor result(resultType);
  for (auto it = result.index_begin(); it != result.index_end(); ++it) {
    Element minElement = min.getRank() != 0 ? min.get(*it) : min.get({});
    Element maxElement = max.getRank() != 0 ? max.get(*it) : max.get({});
    result.set(*it, stablehlo::min(stablehlo::max(operand.get(*it), minElement),
                                   maxElement));
  }
  return result;
}

Tensor evalCompareOp(const Tensor &lhs, const Tensor &rhs,
                     ComparisonDirection comparisonDirection,
                     ShapedType resultType) {
  Tensor result(resultType);
  auto elementTy = lhs.getElementType();

  if (isSupportedComplexType(elementTy) &&
      (comparisonDirection == ComparisonDirection::GE ||
       comparisonDirection == ComparisonDirection::GT ||
       comparisonDirection == ComparisonDirection::LE ||
       comparisonDirection == ComparisonDirection::LT)) {
    report_fatal_error(invalidArgument(
        "Unsupported element type %s for comparison direction %s",
        debugString(elementTy).c_str(),
        debugString(comparisonDirection).c_str()));
  }

  for (auto it = result.index_begin(); it != result.index_end(); ++it) {
    switch (comparisonDirection) {
      case ComparisonDirection::EQ:
        result.set(*it, lhs.get(*it) == rhs.get(*it));
        break;
      case ComparisonDirection::NE:
        result.set(*it, lhs.get(*it) != rhs.get(*it));
        break;
      case ComparisonDirection::GE:
        result.set(*it, lhs.get(*it) >= rhs.get(*it));
        break;
      case ComparisonDirection::GT:
        result.set(*it, lhs.get(*it) > rhs.get(*it));
        break;
      case ComparisonDirection::LE:
        result.set(*it, lhs.get(*it) <= rhs.get(*it));
        break;
      case ComparisonDirection::LT:
        result.set(*it, lhs.get(*it) < rhs.get(*it));
        break;
    }
  }
  return result;
}

Tensor evalConcatenateOp(ArrayRef<Tensor> inputs, Axis dimension,
                         ShapedType resultType) {
  Tensor result(resultType);
  int64_t dimensionOffset = 0;
  for (const auto &input : inputs) {
    for (auto inputIt = input.index_begin(); inputIt != input.index_end();
         ++inputIt) {
      Index resultIdx(*inputIt);
      resultIdx[dimension] += dimensionOffset;
      result.set(resultIdx, input.get(*inputIt));
    }
    dimensionOffset += input.getShape()[dimension];
  }
  return result;
}

Tensor evalConstantOp(ElementsAttr value) {
  return makeTensor(value.cast<DenseElementsAttr>());
}

// This is an simplified implementation of convert op semantics dealing only
// with integer to bool conversion. To be updated as part of #969.
Tensor evalConvertOp(const Tensor &operand, ShapedType resultType) {
  Tensor result(resultType);
  Type elementType = result.getElementType();
  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it, Element(elementType,
                            operand.get(*it).getIntegerValue().getBoolValue()));
  return result;
}

Tensor evalCosineOp(const Tensor &operand, ShapedType resultType) {
  Tensor result(resultType);
  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it, cosine(operand.get(*it)));
  return result;
}

Tensor evalClzOp(const Tensor &operand, ShapedType resultType) {
  Tensor result(resultType);
  for (auto resultIt = result.index_begin(); resultIt != result.index_end();
       ++resultIt) {
    auto element = Element(
        resultType.getElementType(),
        static_cast<int64_t>(
            operand.get(*resultIt).getIntegerValue().countLeadingZeros()));
    result.set(*resultIt, element);
  }
  return result;
}

Tensor evalDivideOp(const Tensor &lhs, const Tensor &rhs,
                    ShapedType resultType) {
  Tensor result(resultType);
  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it, lhs.get(*it) / rhs.get(*it));
  return result;
}

Tensor evalDynamicSliceOp(const Tensor &operand, ArrayRef<Tensor> startIndices,
                          Sizes sliceSizes, ShapedType resultType) {
  Tensor result(resultType);
  auto adjustedStartIndices =
      clamp(0, evalIndices(startIndices), operand.getShape() - sliceSizes);
  for (auto resultIt = result.index_begin(); resultIt != result.index_end();
       ++resultIt) {
    result.set(*resultIt, operand.get(adjustedStartIndices + *resultIt));
  }
  return result;
}

Tensor evalDynamicUpdateSliceOp(const Tensor &operand, const Tensor &update,
                                ArrayRef<Tensor> startIndices,
                                ShapedType resultType) {
  Tensor result(resultType);
  auto adjustedStartIndices = clamp(0, evalIndices(startIndices),
                                    operand.getShape() - update.getShape());
  for (auto resultIt = result.index_begin(); resultIt != result.index_end();
       ++resultIt)
    result.set(*resultIt, operand.get(*resultIt));
  for (auto updateIt = update.index_begin(); updateIt != update.index_end();
       ++updateIt)
    result.set(*updateIt + adjustedStartIndices, update.get(*updateIt));
  return result;
}

Tensor evalExponentialOp(const Tensor &operand, ShapedType resultType) {
  Tensor result(resultType);
  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it, exponential(operand.get(*it)));
  return result;
}

Tensor evalFloorOp(const Tensor &operand, ShapedType resultType) {
  Tensor result(resultType);
  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it, floor(operand.get(*it)));
  return result;
}

Tensor evalGetDimensionSizeOp(const Tensor &operand, Axis dimension,
                              ShapedType resultType) {
  Tensor result(resultType);
  result.set(
      {}, Element(resultType.getElementType(), operand.getShape()[dimension]));
  return result;
}

SmallVector<Tensor> evalIfOp(const Tensor &pred, Region &trueBranch,
                             Region &falseBranch, Scope &scope) {
  return pred.get({}).getBooleanValue() ? eval(trueBranch, {}, &scope)
                                        : eval(falseBranch, {}, &scope);
}

Tensor evalImagOp(const Tensor &operand, ShapedType resultType) {
  Tensor result(resultType);
  for (auto it = operand.index_begin(); it != operand.index_end(); ++it)
    result.set(*it, imag(operand.get(*it)));
  return result;
}

Tensor evalIotaOp(Axis iotaDimension, ShapedType resultType) {
  Tensor result(resultType);
  Type elementType = result.getElementType();
  for (auto it = result.index_begin(); it != result.index_end(); ++it) {
    if (isSupportedIntegerType(elementType)) {
      result.set(*it, Element(elementType, (*it)[iotaDimension]));
    } else if (isSupportedFloatType(elementType)) {
      result.set(
          *it, Element(elementType, static_cast<double>((*it)[iotaDimension])));
    } else if (isSupportedComplexType(elementType)) {
      result.set(*it,
                 Element(elementType,
                         std::complex<double>(
                             static_cast<double>((*it)[iotaDimension]), 0.0)));
    } else {
      report_fatal_error(invalidArgument("Unsupported element type: %s",
                                         debugString(elementType).c_str()));
    }
  }
  return result;
}

Tensor evalIsFiniteOp(const Tensor &operand, ShapedType resultType) {
  Tensor result(resultType);
  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it, isFinite(operand.get(*it)));
  return result;
}

Tensor evalLogOp(const Tensor &operand, ShapedType resultType) {
  Tensor result(resultType);
  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it, log(operand.get(*it)));
  return result;
}

Tensor evalLogisticOp(const Tensor &operand, ShapedType resultType) {
  Tensor result(resultType);
  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it, logistic(operand.get(*it)));
  return result;
}

Tensor evalMapOp(ArrayRef<Tensor> inputs, Region &computation, Scope &scope,
                 ShapedType resultType) {
  Tensor result(resultType);
  for (auto resultIt = inputs[0].index_begin();
       resultIt != inputs[0].index_end(); ++resultIt) {
    SmallVector<Tensor> args;
    for (size_t i = 0; i < inputs.size(); ++i) {
      auto tensor =
          Tensor(computation.getArgument(i).getType().cast<ShapedType>());
      tensor.set({}, inputs[i].get(*resultIt));
      args.push_back(tensor);
    }
    result.set(*resultIt, eval(computation, args, &scope)[0].get({}));
  }
  return result;
}

Tensor evalMaxOp(const Tensor &lhs, const Tensor &rhs, ShapedType resultType) {
  Tensor result(resultType);
  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it, max(lhs.get(*it), rhs.get(*it)));
  return result;
}

Tensor evalMinOp(const Tensor &lhs, const Tensor &rhs, ShapedType resultType) {
  Tensor result(resultType);
  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it, min(lhs.get(*it), rhs.get(*it)));
  return result;
}

Tensor evalMultiplyOp(const Tensor &lhs, const Tensor &rhs,
                      ShapedType resultType) {
  Tensor result(resultType);
  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it, lhs.get(*it) * rhs.get(*it));
  return result;
}

Tensor evalNegOp(const Tensor &operand, ShapedType resultType) {
  Tensor result(resultType);
  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it, -operand.get(*it));
  return result;
}

Tensor evalNotOp(const Tensor &operand, ShapedType resultType) {
  Tensor result(resultType);
  for (auto it = operand.index_begin(); it != operand.index_end(); ++it)
    result.set(*it, ~operand.get(*it));
  return result;
}

Tensor evalOrOp(const Tensor &lhs, const Tensor &rhs, ShapedType resultType) {
  Tensor result(resultType);
  for (auto it = lhs.index_begin(); it != lhs.index_end(); ++it)
    result.set(*it, lhs.get(*it) | rhs.get(*it));
  return result;
}

Tensor evalPadOp(const Tensor &operand, const Tensor &paddingValue,
                 Sizes edgePaddingLow, Sizes interiorPadding,
                 ShapedType resultType) {
  Tensor result(resultType);
  for (auto resultIt = result.index_begin(); resultIt != result.index_end();
       ++resultIt)
    result.set(*resultIt, paddingValue.get({}));
  for (auto operandIt = operand.index_begin(); operandIt != operand.index_end();
       ++operandIt) {
    auto resultIdx = edgePaddingLow + *operandIt * (interiorPadding + 1);
    // Bound check is needed here because of negative padding which could
    // swallow some operand indices.
    if (resultIdx.inBounds(result.getShape()))
      result.set(resultIdx, operand.get(*operandIt));
  }
  return result;
}

Tensor evalPopulationCountOp(const Tensor &operand, ShapedType resultType) {
  Tensor result(resultType);
  for (auto resultIt = result.index_begin(); resultIt != result.index_end();
       ++resultIt)
    result.set(*resultIt, popcnt(operand.get(*resultIt)));
  return result;
}

Tensor evalPowerOp(const Tensor &lhs, const Tensor &rhs,
                   ShapedType resultType) {
  Tensor result(resultType);
  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it, power(lhs.get(*it), rhs.get(*it)));
  return result;
}

Tensor evalRealOp(const Tensor &operand, ShapedType resultType) {
  Tensor result(resultType);
  for (auto it = operand.index_begin(); it != operand.index_end(); ++it)
    result.set(*it, real(operand.get(*it)));
  return result;
}

Tensor evalRemOp(const Tensor &lhs, const Tensor &rhs, ShapedType resultType) {
  Tensor result(resultType);
  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it, rem(lhs.get(*it), rhs.get(*it)));
  return result;
}

Tensor evalReshapeOp(const Tensor &operand, ShapedType resultType) {
  Tensor result(resultType);
  for (auto resultIt = result.index_begin(), operandIt = operand.index_begin();
       resultIt != result.index_end(); ++resultIt, ++operandIt)
    result.set(*resultIt, operand.get(*operandIt));
  return result;
}

Tensor evalReverseOp(const Tensor &operand, Axes dimensions,
                     ShapedType resultType) {
  Tensor result(resultType);
  auto resultShape = result.getShape();
  for (auto resultIt = result.index_begin(); resultIt != result.index_end();
       ++resultIt) {
    Index operandIdx(*resultIt);
    for (auto dim : dimensions)
      operandIdx[dim] = (resultShape[dim] - 1) - operandIdx[dim];
    result.set(*resultIt, operand.get(operandIdx));
  }
  return result;
}

Tensor evalRoundOp(const Tensor &operand, ShapedType resultType) {
  Tensor result(resultType);
  for (auto resultIt = result.index_begin(); resultIt != result.index_end();
       ++resultIt)
    result.set(*resultIt, roundNearestAfz(operand.get(*resultIt)));
  return result;
}

Tensor evalRoundNearestEvenOp(const Tensor &operand, ShapedType resultType) {
  Tensor result(resultType);
  for (auto resultIt = result.index_begin(); resultIt != result.index_end();
       ++resultIt)
    result.set(*resultIt, roundNearestEven(operand.get(*resultIt)));
  return result;
}

Tensor evalRsqrtOp(const Tensor &operand, ShapedType resultType) {
  Tensor result(resultType);
  for (auto resultIt = result.index_begin(); resultIt != result.index_end();
       ++resultIt)
    result.set(*resultIt, rsqrt(operand.get(*resultIt)));
  return result;
}

Tensor evalSelectOp(const Tensor &pred, const Tensor &onTrue,
                    const Tensor &onFalse, ShapedType resultType) {
  Tensor result(resultType);
  for (auto it = result.index_begin(); it != result.index_end(); ++it) {
    Element predValue = pred.getRank() != 0 ? pred.get(*it) : pred.get({});
    result.set(
        *it, predValue.getBooleanValue() ? onTrue.get(*it) : onFalse.get(*it));
  }
  return result;
}

Tensor evalShiftLeftOp(const Tensor &lhs, const Tensor &rhs,
                       ShapedType resultType) {
  Tensor result(resultType);
  for (auto resultIt = result.index_begin(); resultIt != result.index_end();
       ++resultIt)
    result.set(*resultIt, shiftLeft(lhs.get(*resultIt), rhs.get(*resultIt)));
  return result;
}

Tensor evalShiftRightArithmeticOp(const Tensor &lhs, const Tensor &rhs,
                                  ShapedType resultType) {
  Tensor result(resultType);
  for (auto resultIt = result.index_begin(); resultIt != result.index_end();
       ++resultIt)
    result.set(*resultIt,
               shiftRightArithmetic(lhs.get(*resultIt), rhs.get(*resultIt)));
  return result;
}

Tensor evalShiftRightLogicalOp(const Tensor &lhs, const Tensor &rhs,
                               ShapedType resultType) {
  Tensor result(resultType);
  for (auto resultIt = result.index_begin(); resultIt != result.index_end();
       ++resultIt)
    result.set(*resultIt,
               shiftRightLogical(lhs.get(*resultIt), rhs.get(*resultIt)));
  return result;
}

Tensor evalSignOp(const Tensor &operand, ShapedType resultType) {
  Tensor result(resultType);
  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it, sign(operand.get(*it)));
  return result;
}

Tensor evalSineOp(const Tensor &operand, ShapedType resultType) {
  Tensor result(resultType);
  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it, sine(operand.get(*it)));
  return result;
}

Tensor evalSliceOp(const Tensor &operand, Index startIndices, Sizes strides,
                   ShapedType resultType) {
  Tensor result(resultType);
  for (auto resultIt = result.index_begin(); resultIt != result.index_end();
       ++resultIt) {
    result.set(*resultIt, operand.get(startIndices + *resultIt * strides));
  }
  return result;
}

SmallVector<Tensor> evalSortOp(ArrayRef<Tensor> inputs, Axis dimension,
                               bool isStable, Region &comparator,
                               Scope &scope) {
  SmallVector<Tensor> results;
  for (auto input : inputs) results.push_back(Tensor(input.getType()));
  auto adjustedDimension =
      dimension >= 0 ? dimension : dimension + inputs[0].getRank();

  for (auto resultIt = results[0].index_begin();
       resultIt != results[0].index_end(); ++resultIt) {
    // resultIt iterates through all indices in the index space, but sorting
    // only needs to be done once per slice.
    if ((*resultIt)[adjustedDimension] != 0) continue;

    // Instead of literally putting the slices together into a vector of tuples,
    // we're representing these tuples with integer handles, with each handle
    // being an index within the slice.
    // Then, instead of sorting a vector of tuples, we're sorting a vector of
    // handles, and the comparator knows how to use these handles to fetch
    // the actual input elements being compared.
    Index inputsTogether(inputs[0].getShape()[adjustedDimension]);
    std::iota(inputsTogether.begin(), inputsTogether.end(), 0);
    auto comparatorTogether = [&](int64_t lhsHandle, int64_t rhsHandle) {
      SmallVector<Tensor> args;
      auto lhsIndex = *resultIt;
      auto rhsIndex = *resultIt;
      lhsIndex[adjustedDimension] = lhsHandle;
      rhsIndex[adjustedDimension] = rhsHandle;
      for (const auto &input : inputs) {
        auto argType = RankedTensorType::get({}, input.getElementType());
        auto lhsEl = Tensor(argType);
        auto rhsEl = Tensor(argType);
        lhsEl.set({}, input.get(lhsIndex));
        rhsEl.set({}, input.get(rhsIndex));
        args.push_back(lhsEl);
        args.push_back(rhsEl);
      }
      auto cmpResult = eval(comparator, args, &scope);
      return cmpResult[0].get({}).getBooleanValue();
    };
    if (isStable)
      std::stable_sort(inputsTogether.begin(), inputsTogether.end(),
                       comparatorTogether);
    else
      std::sort(inputsTogether.begin(), inputsTogether.end(),
                comparatorTogether);

    // After the vector of handles has been sorted, we apply the results of
    // this sort by reshuffling input elements into result elements.
    for (auto [inputHandle, resultHandle] : llvm::enumerate(inputsTogether)) {
      for (auto [input, result] : llvm::zip(inputs, results)) {
        auto inputIdx = *resultIt;
        auto resultIdx = *resultIt;
        inputIdx[adjustedDimension] = inputHandle;
        resultIdx[adjustedDimension] = resultHandle;
        result.set(resultIdx, input.get(inputIdx));
      }
    }
  }
  return results;
}

Tensor evalSqrtOp(const Tensor &operand, ShapedType resultType) {
  Tensor result(resultType);
  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it, sqrt(operand.get(*it)));
  return result;
}

Tensor evalSubtractOp(const Tensor &lhs, const Tensor &rhs,
                      ShapedType resultType) {
  Tensor result(resultType);
  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it, lhs.get(*it) - rhs.get(*it));
  return result;
}

Tensor evalTanhOp(const Tensor &operand, ShapedType resultType) {
  Tensor result(resultType);
  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it, tanh(operand.get(*it)));
  return result;
}

Tensor evalTransposeOp(const Tensor &operand, const Axes &permutation,
                       ShapedType resultType) {
  Tensor result(resultType);
  for (auto operandIt = operand.index_begin(); operandIt != operand.index_end();
       ++operandIt) {
    auto resultIdx = operandIt->permute(permutation);
    result.set(resultIdx, operand.get(*operandIt));
  }
  return result;
}

SmallVector<Tensor> evalWhileOp(ArrayRef<Tensor> operand, Region &cond,
                                Region &body, Scope &scope) {
  SmallVector<Tensor> runtimeResults(operand);

  auto condResults = eval(cond, operand, &scope);
  if (condResults.size() != 1)
    llvm::report_fatal_error("Failed to evaluate cond");

  while (condResults[0].get(*condResults[0].index_begin()).getBooleanValue()) {
    runtimeResults = eval(body, runtimeResults, &scope);
    condResults = eval(cond, runtimeResults, &scope);
    if (condResults.size() != 1)
      llvm::report_fatal_error("Failed to evaluate cond");
  }

  return runtimeResults;
}

Tensor evalXorOp(const Tensor &lhs, const Tensor &rhs, ShapedType resultType) {
  Tensor result(resultType);
  for (auto it = lhs.index_begin(); it != lhs.index_end(); ++it)
    result.set(*it, lhs.get(*it) ^ rhs.get(*it));
  return result;
}

}  // namespace stablehlo
}  // namespace mlir
