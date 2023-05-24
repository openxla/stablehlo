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
#include "stablehlo/dialect/TypeInference.h"
#include "stablehlo/reference/Element.h"
#include "stablehlo/reference/Errors.h"
#include "stablehlo/reference/Types.h"

namespace mlir {
namespace stablehlo {
namespace {

Index evalIndices(ArrayRef<Tensor> indices) {
  Index index(indices.size());
  for (size_t i = 0; i < indices.size(); ++i)
    index[i] = indices[i].get({}).getIntegerValue().getSExtValue();
  return index;
}

Tensor evalPadOp(const Tensor &operand, const Tensor &paddingValue,
                 const Sizes &edgePaddingLow, const Sizes &edgePaddingHigh,
                 const Sizes &interiorPadding) {
  SmallVector<Type> inferredTypes;
  Builder builder(operand.getType().getContext());
  auto inferStatus =
      hlo::inferPadOp({}, operand.getType(), paddingValue.getType(),
                      builder.getI64TensorAttr(edgePaddingLow),
                      builder.getI64TensorAttr(edgePaddingHigh),
                      builder.getI64TensorAttr(interiorPadding), inferredTypes);
  if (failed(inferStatus))
    report_fatal_error(invalidArgument("Could not infer PadOp's return type"));
  return evalPadOp(operand, paddingValue, edgePaddingLow, interiorPadding,
                   inferredTypes[0].cast<ShapedType>());
}

SmallVector<Tensor> evalReduceOp(ArrayRef<Tensor> inputs,
                                 ArrayRef<Tensor> initValues,
                                 const Axes &dimensions, Region &body,
                                 Scope &scope) {
  SmallVector<Type> inputTypes;
  for (const auto &input : inputs) inputTypes.push_back(input.getType());

  SmallVector<Type> initValueTypes;
  for (const auto &initValue : initValues)
    initValueTypes.push_back(initValue.getType());

  SmallVector<ShapedTypeComponents> inferredReduceTypes;
  Builder builder(inputs[0].getType().getContext());
  auto reduceStatus = hlo::inferReduceOp(
      /*location=*/{}, inputTypes, initValueTypes,
      builder.getI64TensorAttr(dimensions), inferredReduceTypes);
  if (failed(reduceStatus))
    report_fatal_error(
        invalidArgument("Could not infer ReduceOp's return type"));

  SmallVector<ShapedType> resultTypes;
  for (const auto &inferredType : inferredReduceTypes) {
    auto shapedType = hlo::createShapedType(inferredType);
    if (!shapedType)
      llvm::report_fatal_error("Could not infer ReduceOp's return type");
    resultTypes.push_back(shapedType);
  }
  return evalReduceOp(inputs, initValues, dimensions, body, scope, resultTypes);
}

Tensor evalSliceOp(const Tensor &operand, const Sizes &startIndices,
                   const Sizes &limitIndices, const Sizes &strides) {
  SmallVector<Type> inferredTypes;
  Builder builder(operand.getType().getContext());
  auto inferStatus = hlo::inferSliceOp(
      {}, operand.getType(), builder.getI64TensorAttr(startIndices),
      builder.getI64TensorAttr(limitIndices), builder.getI64TensorAttr(strides),
      inferredTypes);
  if (failed(inferStatus))
    report_fatal_error(
        invalidArgument("Could not infer SliceOp's return type"));
  return evalSliceOp(operand, startIndices, strides,
                     inferredTypes[0].cast<ShapedType>());
}

Tensor computeSum(const Tensor &input, const Tensor &initValue,
                  const Axes &dimensions, ShapedType resultType) {
  Tensor result(resultType, initValue.get({}));
  for (auto inputIt = input.index_begin(); inputIt != input.index_end();
       ++inputIt) {
    Index resultIndex;
    for (auto [inputAxis, inputIndexElement] : llvm::enumerate(*inputIt)) {
      if (llvm::is_contained(dimensions, inputAxis)) continue;
      resultIndex.push_back(inputIndexElement);
    }
    result.set(resultIndex, result.get(resultIndex) + input.get(*inputIt));
  }
  return result;
}

Tensor computeMean(const Tensor &operand, Axis featureIndex,
                   ShapedType resultType) {
  auto dimensions = operand.getAxes();
  dimensions.erase(dimensions.begin() + featureIndex);

  auto sum =
      computeSum(operand,
                 Tensor(RankedTensorType::get({}, operand.getElementType()),
                        convert(operand.getElementType(), 0.0)),
                 dimensions, resultType);

  auto divisor = Tensor(RankedTensorType::get({}, operand.getElementType()),
                        convert(operand.getElementType(),
                                static_cast<double>(operand.getNumElements()) /
                                    operand.getShape()[featureIndex]));
  auto divisorBroadcast = evalBroadcastInDimOp(divisor, {}, sum.getType());

  return evalDivideOp(sum, divisorBroadcast, sum.getType());
}

Tensor computeVariance(const Tensor &operand, Axis featureIndex,
                       ShapedType resultType) {
  auto mean = computeMean(operand, featureIndex, resultType);
  auto meanBroadcast =
      evalBroadcastInDimOp(mean, {featureIndex}, operand.getType());
  auto centeredOperand =
      evalSubtractOp(operand, meanBroadcast, operand.getType());
  return computeMean(evalMultiplyOp(centeredOperand, centeredOperand,
                                    centeredOperand.getType()),
                     featureIndex, resultType);
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
      auto operand = scope.find(absOp.getOperand());
      auto result = evalAbsOp(operand, absOp.getType());
      scope.add(op.getResults(), {result});
    } else if (auto addOp = dyn_cast<AddOp>(op)) {
      auto lhs = scope.find(addOp.getLhs());
      auto rhs = scope.find(addOp.getRhs());
      auto result = evalAddOp(lhs, rhs, addOp.getType());
      scope.add(op.getResults(), {result});
    } else if (auto andOp = dyn_cast<AndOp>(op)) {
      auto lhs = scope.find(andOp.getLhs());
      auto rhs = scope.find(andOp.getRhs());
      auto result = evalAndOp(lhs, rhs, andOp.getType());
      scope.add(op.getResults(), {result});
    } else if (auto atan2Op = dyn_cast<Atan2Op>(op)) {
      auto lhs = scope.find(atan2Op.getLhs());
      auto rhs = scope.find(atan2Op.getRhs());
      auto result = evalAtan2Op(lhs, rhs, atan2Op.getType());
      scope.add(op.getResults(), {result});
    } else if (auto batchNormInferenceOp = dyn_cast<BatchNormInferenceOp>(op)) {
      auto operand = scope.find(batchNormInferenceOp.getOperand());
      auto scale = scope.find(batchNormInferenceOp.getScale());
      auto offset = scope.find(batchNormInferenceOp.getOffset());
      auto mean = scope.find(batchNormInferenceOp.getMean());
      auto variance = scope.find(batchNormInferenceOp.getVariance());
      auto result =
          evalBatchNormInferenceOp(operand, scale, offset, mean, variance,
                                   batchNormInferenceOp.getEpsilon(),
                                   batchNormInferenceOp.getFeatureIndex(),
                                   batchNormInferenceOp.getType());
      scope.add(op.getResults(), {result});
    } else if (auto batchNormTrainingOp = dyn_cast<BatchNormTrainingOp>(op)) {
      auto operand = scope.find(batchNormTrainingOp.getOperand());
      auto scale = scope.find(batchNormTrainingOp.getScale());
      auto offset = scope.find(batchNormTrainingOp.getOffset());
      auto results = evalBatchNormTrainingOp(
          operand, scale, offset, batchNormTrainingOp.getEpsilon(),
          batchNormTrainingOp.getFeatureIndex(),
          {batchNormTrainingOp.getOutput().getType(),
           batchNormTrainingOp.getBatchMean().getType(),
           batchNormTrainingOp.getBatchVar().getType()});
      scope.add(op.getResults(), results);
    } else if (auto broadcastInDimOp = dyn_cast<BroadcastInDimOp>(op)) {
      auto operand = scope.find(broadcastInDimOp.getOperand());
      auto broadcastDimensions =
          Axes(broadcastInDimOp.getBroadcastDimensions());
      auto result = evalBroadcastInDimOp(operand, broadcastDimensions,
                                         broadcastInDimOp.getType());
      scope.add(op.getResults(), {result});
    } else if (auto caseOp = dyn_cast<CaseOp>(op)) {
      auto index = scope.find(caseOp.getIndex());
      auto branches = caseOp.getBranches();
      auto results = evalCaseOp(index, branches, scope);
      scope.add(op.getResults(), results);
    } else if (auto cbrtOp = dyn_cast<CbrtOp>(op)) {
      auto operand = scope.find(cbrtOp.getOperand());
      auto result = evalCbrtOp(operand, cbrtOp.getType());
      scope.add(op.getResults(), {result});
    } else if (auto ceilOp = dyn_cast<CeilOp>(op)) {
      auto operand = scope.find(ceilOp.getOperand());
      auto result = evalCeilOp(operand, ceilOp.getType());
      scope.add(op.getResults(), {result});
    } else if (auto choleskyOp = dyn_cast<CholeskyOp>(op)) {
      auto a = scope.find(choleskyOp.getA());
      auto result =
          evalCholeskyOp(a, choleskyOp.getLower(), choleskyOp.getType());
      scope.add(op.getResults(), {result});
    } else if (auto clampOp = dyn_cast<ClampOp>(op)) {
      auto min = scope.find(clampOp.getMin());
      auto operand = scope.find(clampOp.getOperand());
      auto max = scope.find(clampOp.getMax());
      auto result = evalClampOp(min, operand, max, clampOp.getType());
      scope.add(op.getResults(), {result});
    } else if (auto clzOp = dyn_cast<ClzOp>(op)) {
      auto operand = scope.find(clzOp.getOperand());
      auto result = evalClzOp(operand, clzOp.getType());
      scope.add(op.getResults(), {result});
    } else if (auto compareOp = dyn_cast<CompareOp>(op)) {
      auto lhs = scope.find(compareOp.getLhs());
      auto rhs = scope.find(compareOp.getRhs());
      auto comparisonDirection = compareOp.getComparisonDirection();
      auto result =
          evalCompareOp(lhs, rhs, comparisonDirection, compareOp.getType());
      scope.add(op.getResults(), {result});
    } else if (auto complexOp = dyn_cast<ComplexOp>(op)) {
      auto lhs = scope.find(complexOp.getLhs());
      auto rhs = scope.find(complexOp.getRhs());
      auto result = evalComplexOp(lhs, rhs, complexOp.getType());
      scope.add(op.getResults(), {result});
    } else if (auto concatenateOp = dyn_cast<ConcatenateOp>(op)) {
      auto operands = scope.find(concatenateOp.getOperands());
      auto result = evalConcatenateOp(operands, concatenateOp.getDimension(),
                                      concatenateOp.getType());
      scope.add(op.getResults(), {result});
    } else if (auto constantOp = dyn_cast<ConstantOp>(op)) {
      auto result = evalConstantOp(constantOp.getValue());
      scope.add(op.getResults(), {result});
    } else if (auto convertOp = dyn_cast<ConvertOp>(op)) {
      auto operand = scope.find(convertOp.getOperand());
      auto result = evalConvertOp(operand, convertOp.getType());
      scope.add(op.getResults(), {result});
    } else if (auto cosineOp = dyn_cast<CosineOp>(op)) {
      auto operand = scope.find(cosineOp.getOperand());
      auto result = evalCosineOp(operand, cosineOp.getType());
      scope.add(op.getResults(), {result});
    } else if (auto divideOp = dyn_cast<DivOp>(op)) {
      auto lhs = scope.find(divideOp.getLhs());
      auto rhs = scope.find(divideOp.getRhs());
      auto result = evalDivideOp(lhs, rhs, divideOp.getType());
      scope.add(op.getResults(), {result});
    } else if (auto dynamicSliceOp = dyn_cast<DynamicSliceOp>(op)) {
      auto operand = scope.find(dynamicSliceOp.getOperand());
      auto startIndices = scope.find(dynamicSliceOp.getStartIndices());
      auto sliceSizes = Sizes(dynamicSliceOp.getSliceSizes());
      auto result = evalDynamicSliceOp(operand, startIndices, sliceSizes,
                                       dynamicSliceOp.getType());
      scope.add(op.getResults(), {result});
    } else if (auto dynamicUpdateSliceOp = dyn_cast<DynamicUpdateSliceOp>(op)) {
      auto operand = scope.find(dynamicUpdateSliceOp.getOperand());
      auto update = scope.find(dynamicUpdateSliceOp.getUpdate());
      auto startIndices = scope.find(dynamicUpdateSliceOp.getStartIndices());
      auto result = evalDynamicUpdateSliceOp(operand, update, startIndices,
                                             dynamicUpdateSliceOp.getType());
      scope.add(op.getResults(), {result});
    } else if (auto expOp = dyn_cast<ExpOp>(op)) {
      auto operand = scope.find(expOp.getOperand());
      auto result = evalExponentialOp(operand, expOp.getType());
      scope.add(op.getResults(), {result});
    } else if (auto expm1Op = dyn_cast<Expm1Op>(op)) {
      auto operand = scope.find(expm1Op.getOperand());
      auto result = evalExpm1Op(operand, expm1Op.getType());
      scope.add(op.getResults(), {result});
    } else if (auto floorOp = dyn_cast<FloorOp>(op)) {
      auto operand = scope.find(floorOp.getOperand());
      auto result = evalFloorOp(operand, floorOp.getType());
      scope.add(op.getResults(), {result});
    } else if (auto getDimensionSizeOp = dyn_cast<GetDimensionSizeOp>(op)) {
      auto operand = scope.find(getDimensionSizeOp.getOperand());
      auto dimension = getDimensionSizeOp.getDimension();
      auto results = evalGetDimensionSizeOp(operand, dimension,
                                            getDimensionSizeOp.getType());
      scope.add(op.getResults(), results);
    } else if (auto ifOp = dyn_cast<IfOp>(op)) {
      auto pred = scope.find(ifOp.getPred());
      auto &trueBranch = ifOp.getTrueBranch();
      auto &falseBranch = ifOp.getFalseBranch();
      auto results = evalIfOp(pred, trueBranch, falseBranch, scope);
      scope.add(op.getResults(), results);
    } else if (auto imagOp = dyn_cast<ImagOp>(op)) {
      auto operand = scope.find(imagOp.getOperand());
      auto result = evalImagOp(operand, imagOp.getType());
      scope.add(op.getResults(), {result});
    } else if (auto iotaOp = dyn_cast<IotaOp>(op)) {
      auto iotaDimension = iotaOp.getIotaDimension();
      auto result = evalIotaOp(iotaDimension, iotaOp.getType());
      scope.add(op.getResults(), {result});
    } else if (auto isFiniteOp = dyn_cast<IsFiniteOp>(op)) {
      auto operand = scope.find(isFiniteOp.getOperand());
      auto result = evalIsFiniteOp(operand, isFiniteOp.getType());
      scope.add(op.getResults(), {result});
    } else if (auto logOp = dyn_cast<LogOp>(op)) {
      auto operand = scope.find(logOp.getOperand());
      auto result = evalLogOp(operand, logOp.getType());
      scope.add(op.getResults(), {result});
    } else if (auto log1pOp = dyn_cast<Log1pOp>(op)) {
      auto operand = scope.find(log1pOp.getOperand());
      auto result = evalLog1pOp(operand, log1pOp.getType());
      scope.add(op.getResults(), {result});
    } else if (auto logisticOp = dyn_cast<LogisticOp>(op)) {
      auto operand = scope.find(logisticOp.getOperand());
      auto result = evalLogisticOp(operand, logisticOp.getType());
      scope.add(op.getResults(), {result});
    } else if (auto mapOp = dyn_cast<MapOp>(op)) {
      auto inputs = scope.find(mapOp.getInputs());
      auto &computation = mapOp.getComputation();
      auto results = evalMapOp(inputs, computation, scope, mapOp.getType());
      scope.add(op.getResults(), results);
    } else if (auto maxOp = dyn_cast<MaxOp>(op)) {
      auto lhs = scope.find(maxOp.getLhs());
      auto rhs = scope.find(maxOp.getRhs());
      auto result = evalMaxOp(lhs, rhs, maxOp.getType());
      scope.add(op.getResults(), {result});
    } else if (auto minOp = dyn_cast<MinOp>(op)) {
      auto lhs = scope.find(minOp.getLhs());
      auto rhs = scope.find(minOp.getRhs());
      auto result = evalMinOp(lhs, rhs, minOp.getType());
      scope.add(op.getResults(), {result});
    } else if (auto multiplyOp = dyn_cast<MulOp>(op)) {
      auto lhs = scope.find(multiplyOp.getLhs());
      auto rhs = scope.find(multiplyOp.getRhs());
      auto result = evalMultiplyOp(lhs, rhs, multiplyOp.getType());
      scope.add(op.getResults(), {result});
    } else if (auto negOp = dyn_cast<NegOp>(op)) {
      auto operand = scope.find(negOp.getOperand());
      auto result = evalNegOp(operand, negOp.getType());
      scope.add(op.getResults(), {result});
    } else if (auto notOp = dyn_cast<NotOp>(op)) {
      auto operand = scope.find(notOp.getOperand());
      auto result = evalNotOp(operand, notOp.getType());
      scope.add(op.getResults(), {result});
    } else if (auto orOp = dyn_cast<OrOp>(op)) {
      auto lhs = scope.find(orOp.getLhs());
      auto rhs = scope.find(orOp.getRhs());
      auto result = evalOrOp(lhs, rhs, orOp.getType());
      scope.add(op.getResults(), {result});
    } else if (auto padOp = dyn_cast<PadOp>(op)) {
      auto operand = scope.find(padOp.getOperand());
      auto paddingValue = scope.find(padOp.getPaddingValue());
      auto edgePaddingLow = Sizes(padOp.getEdgePaddingLow());
      auto interiorPadding = Sizes(padOp.getInteriorPadding());
      auto result = evalPadOp(operand, paddingValue, edgePaddingLow,
                              interiorPadding, padOp.getType());
      scope.add(op.getResults(), {result});
    } else if (auto populationCountOp = dyn_cast<PopulationCountOp>(op)) {
      auto operand = scope.find(populationCountOp.getOperand());
      auto result = evalPopulationCountOp(operand, populationCountOp.getType());
      scope.add(op.getResults(), {result});
    } else if (auto powerOp = dyn_cast<PowOp>(op)) {
      auto lhs = scope.find(powerOp.getLhs());
      auto rhs = scope.find(powerOp.getRhs());
      auto result = evalPowerOp(lhs, rhs, powerOp.getType());
      scope.add(op.getResults(), {result});
    } else if (auto realOp = dyn_cast<RealOp>(op)) {
      auto operand = scope.find(realOp.getOperand());
      auto result = evalRealOp(operand, realOp.getType());
      scope.add(op.getResults(), {result});
    } else if (auto reduceOp = dyn_cast<ReduceOp>(op)) {
      auto inputs = scope.find(reduceOp.getInputs());
      auto initValues = scope.find(reduceOp.getInitValues());
      SmallVector<ShapedType> resultTypes;
      for (auto resultType : reduceOp.getResultTypes())
        resultTypes.push_back(resultType.cast<ShapedType>());
      auto results =
          evalReduceOp(inputs, initValues, Axes(reduceOp.getDimensions()),
                       reduceOp.getBody(), scope, resultTypes);
      scope.add(op.getResults(), results);
    } else if (auto reduceWindowOp = dyn_cast<ReduceWindowOp>(op)) {
      auto inputs = scope.find(reduceWindowOp.getInputs());
      auto initValues = scope.find(reduceWindowOp.getInitValues());
      int64_t rank = inputs[0].getRank();

      Sizes windowStrides(rank, 1);
      if (auto windowStridesAttr = reduceWindowOp.getWindowStridesAttr())
        windowStrides.assign(windowStridesAttr.value_begin<int64_t>(),
                             windowStridesAttr.value_end<int64_t>());

      Sizes baseDilations(rank, 1);
      if (auto baseDilationsAttr = reduceWindowOp.getBaseDilationsAttr())
        baseDilations.assign(baseDilationsAttr.value_begin<int64_t>(),
                             baseDilationsAttr.value_end<int64_t>());

      Sizes windowDilations(rank, 1);
      if (auto windowDilationsAttr = reduceWindowOp.getWindowDilationsAttr())
        windowDilations.assign(windowDilationsAttr.value_begin<int64_t>(),
                               windowDilationsAttr.value_end<int64_t>());

      Sizes paddingLow(rank, 0), paddingHigh(rank, 0);
      if (auto paddingAttr = reduceWindowOp.getPaddingAttr()) {
        auto paddingOrErr =
            hlo::convertPaddingAttribute(reduceWindowOp.getPadding(), {});
        if (failed(paddingOrErr))
          report_fatal_error(invalidArgument("Invalid padding format found."));
        for (auto i = 0; i < static_cast<int64_t>(paddingOrErr->size()); ++i) {
          paddingLow[i] = (*paddingOrErr)[i].first;
          paddingHigh[i] = (*paddingOrErr)[i].second;
        }
      }

      SmallVector<ShapedType> resultTypes;
      for (auto resultType : reduceWindowOp.getResultTypes())
        resultTypes.push_back(resultType.cast<ShapedType>());

      SmallVector<Tensor> results = evalReduceWindowOp(
          inputs, initValues, Sizes(reduceWindowOp.getWindowDimensions()),
          windowStrides, baseDilations, windowDilations, paddingLow,
          paddingHigh, reduceWindowOp.getBody(), scope, resultTypes);
      scope.add(op.getResults(), results);
    } else if (auto remOp = dyn_cast<RemOp>(op)) {
      auto lhs = scope.find(remOp.getLhs());
      auto rhs = scope.find(remOp.getRhs());
      auto result = evalRemOp(lhs, rhs, remOp.getType());
      scope.add(op.getResults(), {result});
    } else if (auto reshapeOp = dyn_cast<ReshapeOp>(op)) {
      auto operand = scope.find(reshapeOp.getOperand());
      auto result = evalReshapeOp(operand, reshapeOp.getType());
      scope.add(op.getResults(), {result});
    } else if (auto returnOp = dyn_cast<func::ReturnOp>(op)) {
      return scope.find(returnOp.getOperands());
    } else if (auto returnOp = dyn_cast<ReturnOp>(op)) {
      return scope.find(returnOp.getResults());
    } else if (auto reverseOp = dyn_cast<ReverseOp>(op)) {
      auto operand = scope.find(reverseOp.getOperand());
      auto dimensions = Axes(reverseOp.getDimensions());
      auto result = evalReverseOp(operand, dimensions, reverseOp.getType());
      scope.add(op.getResults(), {result});
    } else if (auto roundOp = dyn_cast<RoundOp>(op)) {
      auto operand = scope.find(roundOp.getOperand());
      auto result = evalRoundOp(operand, roundOp.getType());
      scope.add(op.getResults(), {result});
    } else if (auto roundNearestEvenOp = dyn_cast<RoundNearestEvenOp>(op)) {
      auto operand = scope.find(roundNearestEvenOp.getOperand());
      auto result =
          evalRoundNearestEvenOp(operand, roundNearestEvenOp.getType());
      scope.add(op.getResults(), {result});
    } else if (auto rsqrtOp = dyn_cast<RsqrtOp>(op)) {
      auto operand = scope.find(rsqrtOp.getOperand());
      auto result = evalRsqrtOp(operand, rsqrtOp.getType());
      scope.add(op.getResults(), {result});
    } else if (auto selectOp = dyn_cast<SelectOp>(op)) {
      auto pred = scope.find(selectOp.getPred());
      auto onTrue = scope.find(selectOp.getOnTrue());
      auto onFalse = scope.find(selectOp.getOnFalse());
      auto result = evalSelectOp(pred, onTrue, onFalse, selectOp.getType());
      scope.add(op.getResults(), {result});
    } else if (auto shiftLeftOp = dyn_cast<ShiftLeftOp>(op)) {
      auto lhs = scope.find(shiftLeftOp.getLhs());
      auto rhs = scope.find(shiftLeftOp.getRhs());
      auto result = evalShiftLeftOp(lhs, rhs, shiftLeftOp.getType());
      scope.add(op.getResults(), {result});
    } else if (auto shiftRightArithmeticOp =
                   dyn_cast<ShiftRightArithmeticOp>(op)) {
      auto lhs = scope.find(shiftRightArithmeticOp.getLhs());
      auto rhs = scope.find(shiftRightArithmeticOp.getRhs());
      auto result = evalShiftRightArithmeticOp(
          lhs, rhs, shiftRightArithmeticOp.getType());
      scope.add(op.getResults(), {result});
    } else if (auto shiftRightLogicalOp = dyn_cast<ShiftRightLogicalOp>(op)) {
      auto lhs = scope.find(shiftRightLogicalOp.getLhs());
      auto rhs = scope.find(shiftRightLogicalOp.getRhs());
      auto result =
          evalShiftRightLogicalOp(lhs, rhs, shiftRightLogicalOp.getType());
      scope.add(op.getResults(), {result});
    } else if (auto signOp = dyn_cast<SignOp>(op)) {
      auto operand = scope.find(signOp.getOperand());
      auto result = evalSignOp(operand, signOp.getType());
      scope.add(op.getResults(), {result});
    } else if (auto sineOp = dyn_cast<SineOp>(op)) {
      auto operand = scope.find(sineOp.getOperand());
      auto result = evalSineOp(operand, sineOp.getType());
      scope.add(op.getResults(), {result});
    } else if (auto sliceOp = dyn_cast<SliceOp>(op)) {
      auto operand = scope.find(sliceOp.getOperand());
      auto startIndices = Sizes(sliceOp.getStartIndices());
      auto strides = Sizes(sliceOp.getStrides());
      auto result =
          evalSliceOp(operand, startIndices, strides, sliceOp.getType());
      scope.add(op.getResults(), {result});
    } else if (auto sortOp = dyn_cast<SortOp>(op)) {
      auto operands = scope.find(sortOp.getInputs());
      auto dimension = sortOp.getDimension();
      auto isStable = sortOp.getIsStable();
      auto &comparator = sortOp.getComparator();
      auto results =
          evalSortOp(operands, dimension, isStable, comparator, scope);
      scope.add(op.getResults(), results);
    } else if (auto sqrtOp = dyn_cast<SqrtOp>(op)) {
      auto operand = scope.find(sqrtOp.getOperand());
      auto result = evalSqrtOp(operand, sqrtOp.getType());
      scope.add(op.getResults(), {result});
    } else if (auto subtractOp = dyn_cast<SubtractOp>(op)) {
      auto lhs = scope.find(subtractOp.getLhs());
      auto rhs = scope.find(subtractOp.getRhs());
      auto result = evalSubtractOp(lhs, rhs, subtractOp.getType());
      scope.add(op.getResults(), {result});
    } else if (auto tanhOp = dyn_cast<TanhOp>(op)) {
      auto operand = scope.find(tanhOp.getOperand());
      auto result = evalTanhOp(operand, tanhOp.getType());
      scope.add(op.getResults(), {result});
    } else if (auto transposeOp = dyn_cast<TransposeOp>(op)) {
      auto operand = scope.find(transposeOp.getOperand());
      auto permutation = Axes(transposeOp.getPermutation());
      auto result =
          evalTransposeOp(operand, permutation, transposeOp.getType());
      scope.add(op.getResults(), {result});
    } else if (auto whileOp = dyn_cast<WhileOp>(op)) {
      auto operand = scope.find(whileOp.getOperand());
      auto &cond = whileOp.getCond();
      auto &body = whileOp.getBody();
      auto results = evalWhileOp(operand, cond, body, scope);
      scope.add(op.getResults(), results);
    } else if (auto xorOp = dyn_cast<XorOp>(op)) {
      auto lhs = scope.find(xorOp.getLhs());
      auto rhs = scope.find(xorOp.getRhs());
      auto result = evalXorOp(lhs, rhs, xorOp.getType());
      scope.add(op.getResults(), {result});
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
  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it, lhs.get(*it) & rhs.get(*it));
  return result;
}

Tensor evalAtan2Op(const Tensor &lhs, const Tensor &rhs,
                   ShapedType resultType) {
  Tensor result(resultType);
  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it, atan2(lhs.get(*it), rhs.get(*it)));
  return result;
}

Tensor evalBatchNormInferenceOp(const Tensor &operand, const Tensor &scale,
                                const Tensor &offset, const Tensor &mean,
                                const Tensor &variance, APFloat epsilon,
                                Axis featureIndex, ShapedType resultType) {
  auto scaleBroadcast =
      evalBroadcastInDimOp(scale, {featureIndex}, operand.getType());
  auto offsetBroadcast =
      evalBroadcastInDimOp(offset, {featureIndex}, operand.getType());
  auto meanBroadcast =
      evalBroadcastInDimOp(mean, {featureIndex}, operand.getType());
  auto varianceBroadcast =
      evalBroadcastInDimOp(variance, {featureIndex}, operand.getType());
  auto epsilonBroadcast = evalBroadcastInDimOp(
      makeTensor(DenseElementsAttr::get(
          RankedTensorType::get({}, operand.getElementType()), {epsilon})),
      {}, operand.getType());

  auto centeredOperand =
      evalSubtractOp(operand, meanBroadcast, operand.getType());
  auto standardDeviation = evalSqrtOp(
      evalAddOp(varianceBroadcast, epsilonBroadcast, operand.getType()),
      operand.getType());
  auto normalizedOperand =
      evalDivideOp(centeredOperand, standardDeviation, operand.getType());

  return evalAddOp(
      evalMultiplyOp(scaleBroadcast, normalizedOperand, operand.getType()),
      offsetBroadcast, operand.getType());
}

SmallVector<Tensor> evalBatchNormTrainingOp(const Tensor &operand,
                                            const Tensor &scale,
                                            const Tensor &offset,
                                            APFloat epsilon, Axis featureIndex,
                                            ArrayRef<ShapedType> resultTypes) {
  auto mean = computeMean(operand, featureIndex, resultTypes[1]);
  auto variance = computeVariance(operand, featureIndex, resultTypes[2]);
  return {evalBatchNormInferenceOp(operand, scale, offset, mean, variance,
                                   epsilon, featureIndex, resultTypes[0]),
          mean, variance};
}

Tensor evalBroadcastInDimOp(const Tensor &operand,
                            const Axes &broadcastDimensions,
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

Tensor evalCbrtOp(const Tensor &operand, ShapedType resultType) {
  Tensor result(resultType);
  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it, cbrt(operand.get(*it)));
  return result;
}

Tensor evalCeilOp(const Tensor &operand, ShapedType resultType) {
  Tensor result(resultType);
  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it, ceil(operand.get(*it)));
  return result;
}

Tensor evalCholeskyOp(const Tensor &a, bool lower, ShapedType resultType) {
  Tensor result(resultType);
  auto aShape = a.getShape();
  auto cholesky = [&lower](const Tensor &A) {
    Tensor L(A.getType());
    auto conjugate = [&](const Element &el) {
      return convert(el.getType(),
                     std::complex<APFloat>(el.getComplexValue().real(),
                                           -el.getComplexValue().imag()));
    };

    for (auto it = L.index_begin(); it != L.index_end(); ++it)
      L.set(*it, convert(A.getElementType(), 0.0));

    for (auto i = 0; i < A.getShape()[0]; ++i) {
      for (auto j = 0; j <= i; ++j) {
        auto sum = convert(A.getElementType(), 0.0);
        for (auto k = 0; k < j; ++k) {
          if (isSupportedComplexType(A.getElementType()))
            sum = sum + L.get(Index({i, k})) * conjugate(L.get(Index({j, k})));
          else
            sum = sum + L.get(Index({i, k})) * L.get(Index({j, k}));
        }
        if (i == j)
          L.set(Index({i, j}), sqrt(A.get(Index({i, i})) - sum));
        else
          L.set(Index({i, j}),
                ((A.get(Index({i, j})) - sum) / L.get(Index({j, j}))));
      }
    }

    if (lower) return L;

    if (isSupportedComplexType(A.getElementType()))
      for (auto it = L.index_begin(); it != L.index_end(); ++it)
        L.set(*it, conjugate(L.get(*it)));
    return evalTransposeOp(L, {1, 0}, L.getType());
  };

  if (a.getRank() == 2) return cholesky(a);

  auto getScalarTensor = [&](auto value) {
    return makeTensor(DenseElementsAttr::get(
        RankedTensorType::get({},
                              IntegerType::get(a.getType().getContext(), 64)),
        {value}));
  };

  Sizes nonBatchingSizes(aShape.end() - 2, aShape.end());
  Sizes batchingSizes(aShape.begin(), aShape.end() - 2);
  for (auto batchIt =
           IndexSpaceIterator(batchingSizes, Sizes(a.getRank() - 2, 0));
       batchIt != IndexSpaceIterator(batchingSizes, std::nullopt); ++batchIt) {
    SmallVector<Tensor> startIndices;
    for (auto index : *batchIt) startIndices.push_back(getScalarTensor(index));
    startIndices.append({getScalarTensor(0L), getScalarTensor(0L)});

    Sizes sliceSizes(a.getRank() - 2, 1);
    sliceSizes.append(nonBatchingSizes);

    auto aSliced = evalDynamicSliceOp(
        a, startIndices, sliceSizes,
        RankedTensorType::get(sliceSizes, a.getElementType()));

    auto L = cholesky(evalReshapeOp(
        aSliced, RankedTensorType::get(nonBatchingSizes, a.getElementType())));

    auto reshapedL =
        evalReshapeOp(L, RankedTensorType::get(sliceSizes, a.getElementType()));

    result =
        evalDynamicUpdateSliceOp(result, reshapedL, startIndices, resultType);
  }
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

Tensor evalComplexOp(const Tensor &lhs, const Tensor &rhs,
                     ShapedType resultType) {
  Tensor result(resultType);
  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it, complex(lhs.get(*it), rhs.get(*it)));
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

Tensor evalConvertOp(const Tensor &operand, ShapedType resultType) {
  Tensor result(resultType);
  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it, convert(result.getElementType(), operand.get(*it)));
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
  for (auto it = result.index_begin(); it != result.index_end(); ++it) {
    auto element =
        convert(resultType.getElementType(),
                static_cast<uint64_t>(
                    operand.get(*it).getIntegerValue().countLeadingZeros()));
    result.set(*it, element);
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
                          const Sizes &sliceSizes, ShapedType resultType) {
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

Tensor evalExpm1Op(const Tensor &operand, ShapedType resultType) {
  Tensor result(resultType);
  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it, exponentialMinusOne(operand.get(*it)));
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
      {}, convert(resultType.getElementType(), operand.getShape()[dimension]));
  return result;
}

SmallVector<Tensor> evalIfOp(const Tensor &pred, Region &trueBranch,
                             Region &falseBranch, Scope &scope) {
  return pred.get({}).getBooleanValue() ? eval(trueBranch, {}, &scope)
                                        : eval(falseBranch, {}, &scope);
}

Tensor evalImagOp(const Tensor &operand, ShapedType resultType) {
  Tensor result(resultType);
  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it, imag(operand.get(*it)));
  return result;
}

Tensor evalIotaOp(Axis iotaDimension, ShapedType resultType) {
  Tensor result(resultType);
  auto elementType = result.getElementType();
  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it, convert(elementType, (*it)[iotaDimension]));
  return result;
}

Tensor evalIsFiniteOp(const Tensor &operand, ShapedType resultType) {
  Tensor result(resultType);
  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it, isFinite(operand.get(*it)));
  return result;
}

Tensor evalLog1pOp(const Tensor &operand, ShapedType resultType) {
  Tensor result(resultType);
  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it, logPlusOne(operand.get(*it)));
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
  for (auto it = result.index_begin(); it != result.index_end(); ++it) {
    SmallVector<Tensor> args;
    for (size_t i = 0; i < inputs.size(); ++i) {
      auto tensor =
          Tensor(computation.getArgument(i).getType().cast<ShapedType>());
      tensor.set({}, inputs[i].get(*it));
      args.push_back(tensor);
    }
    result.set(*it, eval(computation, args, &scope)[0].get({}));
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
  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it, ~operand.get(*it));
  return result;
}

Tensor evalOrOp(const Tensor &lhs, const Tensor &rhs, ShapedType resultType) {
  Tensor result(resultType);
  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it, lhs.get(*it) | rhs.get(*it));
  return result;
}

Tensor evalPadOp(const Tensor &operand, const Tensor &paddingValue,
                 const Sizes &edgePaddingLow, const Sizes &interiorPadding,
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
  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it, popcnt(operand.get(*it)));
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
  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it, real(operand.get(*it)));
  return result;
}

SmallVector<Tensor> evalReduceOp(ArrayRef<Tensor> inputs,
                                 ArrayRef<Tensor> initValues,
                                 const Axes &dimensions, Region &body,
                                 Scope &scope,
                                 ArrayRef<ShapedType> resultTypes) {
  SmallVector<Tensor> results;
  for (auto [resultType, initValue] : llvm::zip(resultTypes, initValues))
    results.push_back(Tensor(resultType, initValue.get({})));

  for (auto inputIt = inputs[0].index_begin(); inputIt != inputs[0].index_end();
       ++inputIt) {
    Index resultIndex;
    for (auto [inputAxis, inputIndexElement] : llvm::enumerate(*inputIt)) {
      if (llvm::is_contained(dimensions, inputAxis)) continue;
      resultIndex.push_back(inputIndexElement);
    }

    SmallVector<Tensor> bodyArgs;
    for (auto [result, initValue] : llvm::zip(results, initValues)) {
      Tensor bodyArg(initValue.getType());
      bodyArg.set({}, result.get(resultIndex));
      bodyArgs.push_back(bodyArg);
    }
    for (auto [input, initValue] : llvm::zip(inputs, initValues)) {
      Tensor bodyArg(initValue.getType());
      bodyArg.set({}, input.get(*inputIt));
      bodyArgs.push_back(bodyArg);
    }

    auto bodyResult = eval(body, bodyArgs, &scope);
    for (auto [result, value] : llvm::zip(results, bodyResult))
      result.set(resultIndex, value.get({}));
  }
  return results;
}

SmallVector<Tensor> evalReduceWindowOp(
    ArrayRef<Tensor> inputs, ArrayRef<Tensor> initValues,
    const Sizes &windowDimensions, const Sizes &windowStrides,
    const Sizes &baseDilations, const Sizes &windowDilations,
    const Sizes &paddingLow, const Sizes &paddingHigh, Region &body,
    Scope &scope, ArrayRef<ShapedType> resultTypes) {
  SmallVector<Tensor> results;
  for (auto [resultType, initValue] : llvm::zip(resultTypes, initValues))
    results.push_back(Tensor(resultType, initValue.get({})));

  SmallVector<Tensor> paddedInputs;
  for (auto [input, initValue] : llvm::zip(inputs, initValues))
    paddedInputs.push_back(evalPadOp(input, initValue, paddingLow, paddingHigh,
                                     baseDilations - 1));

  for (auto resultIt = results[0].index_begin();
       resultIt != results[0].index_end(); ++resultIt) {
    SmallVector<Tensor> windows;
    auto windowStart = (*resultIt) * windowStrides;
    for (const auto &paddedInput : paddedInputs)
      windows.push_back(evalSliceOp(paddedInput, windowStart,
                                    windowStart + windowDimensions,
                                    windowDilations));

    Axes dimensions(inputs[0].getRank());
    std::iota(dimensions.begin(), dimensions.end(), 0);
    auto reducedValues =
        evalReduceOp(windows, initValues, dimensions, body, scope);
    for (auto [result, value] : llvm::zip(results, reducedValues))
      result.set(*resultIt, value.get({}));
  }
  return results;
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

Tensor evalReverseOp(const Tensor &operand, const Axes &dimensions,
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
  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it, roundNearestAfz(operand.get(*it)));
  return result;
}

Tensor evalRoundNearestEvenOp(const Tensor &operand, ShapedType resultType) {
  Tensor result(resultType);
  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it, roundNearestEven(operand.get(*it)));
  return result;
}

Tensor evalRsqrtOp(const Tensor &operand, ShapedType resultType) {
  Tensor result(resultType);
  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it, rsqrt(operand.get(*it)));
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
  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it, shiftLeft(lhs.get(*it), rhs.get(*it)));
  return result;
}

Tensor evalShiftRightArithmeticOp(const Tensor &lhs, const Tensor &rhs,
                                  ShapedType resultType) {
  Tensor result(resultType);
  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it, shiftRightArithmetic(lhs.get(*it), rhs.get(*it)));
  return result;
}

Tensor evalShiftRightLogicalOp(const Tensor &lhs, const Tensor &rhs,
                               ShapedType resultType) {
  Tensor result(resultType);
  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it, shiftRightLogical(lhs.get(*it), rhs.get(*it)));
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

Tensor evalSliceOp(const Tensor &operand, const Sizes &startIndices,
                   const Sizes &strides, ShapedType resultType) {
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
  for (const auto &input : inputs) results.push_back(Tensor(input.getType()));
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
  SmallVector<Tensor> results(operand);

  auto condResults = eval(cond, operand, &scope);
  if (condResults.size() != 1)
    llvm::report_fatal_error("Failed to evaluate cond");

  while (condResults[0].get({}).getBooleanValue()) {
    results = eval(body, results, &scope);
    condResults = eval(cond, results, &scope);
    if (condResults.size() != 1)
      llvm::report_fatal_error("Failed to evaluate cond");
  }

  return results;
}

Tensor evalXorOp(const Tensor &lhs, const Tensor &rhs, ShapedType resultType) {
  Tensor result(resultType);
  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it, lhs.get(*it) ^ rhs.get(*it));
  return result;
}

}  // namespace stablehlo
}  // namespace mlir
