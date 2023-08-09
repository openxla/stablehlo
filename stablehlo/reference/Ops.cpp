/* Copyright 2023 The StableHLO Authors.

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

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/DebugStringHelper.h"
#include "stablehlo/dialect/TypeInference.h"
#include "stablehlo/reference/Axes.h"
#include "stablehlo/reference/Element.h"
#include "stablehlo/reference/Errors.h"
#include "stablehlo/reference/Index.h"
#include "stablehlo/reference/Process.h"
#include "stablehlo/reference/ProcessGrid.h"
#include "stablehlo/reference/Token.h"
#include "stablehlo/reference/Types.h"

namespace mlir {
namespace stablehlo {
namespace {

Index evalIndex(ArrayRef<Tensor> scalars) {
  Index result(scalars.size());
  for (size_t i = 0; i < scalars.size(); ++i)
    result[i] = scalars[i].get({}).getIntegerValue().getSExtValue();
  return result;
}

Index evalIndex(Tensor tensor) {
  Index result;
  for (auto it = tensor.index_begin(); it != tensor.index_end(); ++it)
    result.push_back(tensor.get(*it).getIntegerValue().getSExtValue());
  return result;
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
                                 Process *process, Scope &scope) {
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
  return evalReduceOp(inputs, initValues, dimensions, body, process, scope,
                      resultTypes);
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

// Experimental notation for slices, roughly following the spec notation.
// TODO(#1401): Might evolve in the future together with the spec.
constexpr int64_t kColon = -1;
Tensor evalSliceOp(const Tensor &operand, const Index &index) {
  Sizes start, limit;
  for (auto i = 0; i < operand.getRank(); ++i) {
    if (index[i] == -1) {
      start.push_back(0);
      limit.push_back(operand.getShape()[i]);
    } else {
      start.push_back(index[i]);
      limit.push_back(index[i] + 1);
    }
  }
  Sizes strides(operand.getRank(), 1);
  return evalSliceOp(operand, start, limit, strides);
}

void failOnDecomposableOp(Operation &op) {
  report_fatal_error(invalidArgument(
      "Operation %s is unsupported at the moment. "
      "However, this operation can be decomposed into supported operations, "
      "so it is possible to transform it into supported form as a workaround. "
      "Visit https://github.com/openxla/stablehlo/issues/1571 to learn more "
      "about the workaround and the roadmap for supporting this operation.",
      op.getName().getStringRef().str().c_str()));
}

Tensor makeScalar(const Element &initValue) {
  Tensor result(RankedTensorType::get({}, initValue.getType()));
  result.set({}, initValue);
  return result;
}

Tensor makeSplat(ShapedType type, const Element &initValue) {
  Tensor result(type);
  for (auto indexIt = result.index_begin(); indexIt != result.index_end();
       ++indexIt)
    result.set(*indexIt, initValue);
  return result;
}

}  // namespace

SmallVector<InterpreterValue> eval(
    Region &region, ArrayRef<InterpreterValue> args, Process *process,
    Scope *parent,
    llvm::function_ref<llvm::Error(Operation &, Process *, Scope &)> fallback) {
  Block &block = region.front();
  if (block.getArguments().size() != args.size())
    report_fatal_error(invalidArgument(
        "Expected same number of block arguments and runtime arguments (%d)",
        args.size()));

  Scope scope(parent);
  scope.add(block.getArguments(), args);

  for (Operation &op : block) {
    if (auto absOp = dyn_cast<AbsOp>(op)) {
      auto operand = scope.findTensor(absOp.getOperand());
      auto result = evalAbsOp(operand, absOp.getType());
      scope.add(absOp.getResult(), result);
    } else if (auto addOp = dyn_cast<AddOp>(op)) {
      auto lhs = scope.findTensor(addOp.getLhs());
      auto rhs = scope.findTensor(addOp.getRhs());
      auto result = evalAddOp(lhs, rhs, addOp.getType());
      scope.add(addOp.getResult(), result);
    } else if (auto afterAllOp = dyn_cast<AfterAllOp>(op)) {
      auto inputs = scope.findTokens(afterAllOp.getInputs());
      auto result = evalAfterAllOp(inputs, afterAllOp->getContext());
      scope.add(afterAllOp.getResult(), result);
    } else if (auto allReduceOp = dyn_cast<AllReduceOp>(op)) {
      auto operand = scope.findTensor(allReduceOp.getOperand());

      auto replicaGroupsAttr = allReduceOp.getReplicaGroups();
      auto replicaGroupsShape = replicaGroupsAttr.getShapedType().getShape();
      SmallVector<SmallVector<uint32_t>> replicaGroups(replicaGroupsShape[0]);
      auto replicaGroupsIt = replicaGroupsAttr.getValues<int64_t>().begin();
      for (auto &replicaGroup : replicaGroups) {
        for (auto i = 0; i < replicaGroupsShape[1]; ++i, ++replicaGroupsIt) {
          auto replicaId = *replicaGroupsIt;
          if (replicaId == -1) continue;
          replicaGroup.push_back(replicaId);
        }
      }

      auto channelId = 0;
      if (auto channelHandle = allReduceOp.getChannelHandleAttr())
        channelId = channelHandle.getHandle();

      auto result = evalAllReduceOp(operand, replicaGroups, channelId,
                                    allReduceOp.getUseGlobalDeviceIds(),
                                    allReduceOp.getComputation(), process,
                                    scope, allReduceOp.getType());
      scope.add(allReduceOp.getResult(), result);
    } else if (auto andOp = dyn_cast<AndOp>(op)) {
      auto lhs = scope.findTensor(andOp.getLhs());
      auto rhs = scope.findTensor(andOp.getRhs());
      auto result = evalAndOp(lhs, rhs, andOp.getType());
      scope.add(andOp.getResult(), result);
    } else if (auto atan2Op = dyn_cast<Atan2Op>(op)) {
      auto lhs = scope.findTensor(atan2Op.getLhs());
      auto rhs = scope.findTensor(atan2Op.getRhs());
      auto result = evalAtan2Op(lhs, rhs, atan2Op.getType());
      scope.add(atan2Op.getResult(), result);
    } else if (auto batchNormGradOp = dyn_cast<BatchNormGradOp>(op)) {
      failOnDecomposableOp(op);
    } else if (auto batchNormInferenceOp = dyn_cast<BatchNormInferenceOp>(op)) {
      failOnDecomposableOp(op);
    } else if (auto batchNormTrainingOp = dyn_cast<BatchNormTrainingOp>(op)) {
      failOnDecomposableOp(op);
    } else if (auto bitcastConvertOp = dyn_cast<BitcastConvertOp>(op)) {
      auto operand = scope.findTensor(bitcastConvertOp.getOperand());
      auto result = evalBitcastConvertOp(operand, bitcastConvertOp.getType());
      scope.add(bitcastConvertOp.getResult(), result);
    } else if (auto broadcastInDimOp = dyn_cast<BroadcastInDimOp>(op)) {
      auto operand = scope.findTensor(broadcastInDimOp.getOperand());
      auto broadcastDimensions =
          Axes(broadcastInDimOp.getBroadcastDimensions());
      auto result = evalBroadcastInDimOp(operand, broadcastDimensions,
                                         broadcastInDimOp.getType());
      scope.add(broadcastInDimOp.getResult(), result);
    } else if (isa<BroadcastOp>(op)) {
      failOnDecomposableOp(op);
    } else if (auto caseOp = dyn_cast<CaseOp>(op)) {
      auto index = scope.findTensor(caseOp.getIndex());
      auto branches = caseOp.getBranches();
      auto results = evalCaseOp(index, branches, process, scope);
      scope.add(caseOp.getResults(), results);
    } else if (auto cbrtOp = dyn_cast<CbrtOp>(op)) {
      auto operand = scope.findTensor(cbrtOp.getOperand());
      auto result = evalCbrtOp(operand, cbrtOp.getType());
      scope.add(cbrtOp.getResult(), result);
    } else if (auto ceilOp = dyn_cast<CeilOp>(op)) {
      auto operand = scope.findTensor(ceilOp.getOperand());
      auto result = evalCeilOp(operand, ceilOp.getType());
      scope.add(ceilOp.getResult(), result);
    } else if (auto choleskyOp = dyn_cast<CholeskyOp>(op)) {
      failOnDecomposableOp(op);
    } else if (auto clampOp = dyn_cast<ClampOp>(op)) {
      auto min = scope.findTensor(clampOp.getMin());
      auto operand = scope.findTensor(clampOp.getOperand());
      auto max = scope.findTensor(clampOp.getMax());
      auto result = evalClampOp(min, operand, max, clampOp.getType());
      scope.add(clampOp.getResult(), result);
    } else if (auto clzOp = dyn_cast<ClzOp>(op)) {
      auto operand = scope.findTensor(clzOp.getOperand());
      auto result = evalClzOp(operand, clzOp.getType());
      scope.add(clzOp.getResult(), result);
    } else if (auto collectivePermuteOp = dyn_cast<CollectivePermuteOp>(op)) {
      auto operand = scope.findTensor(collectivePermuteOp.getOperand());

      auto sourceTargetPairsAttr = collectivePermuteOp.getSourceTargetPairs();
      SmallVector<SmallVector<uint32_t>> sourceTargetPairs(
          sourceTargetPairsAttr.getNumElements() / 2);
      auto sourceTargetPairsIt =
          sourceTargetPairsAttr.getValues<int64_t>().begin();
      for (auto &sourceTargetPair : sourceTargetPairs) {
        sourceTargetPair.push_back(*sourceTargetPairsIt++);
        sourceTargetPair.push_back(*sourceTargetPairsIt++);
      }

      ChannelId channelId = 0;
      if (auto channelHandle = collectivePermuteOp.getChannelHandleAttr())
        channelId = channelHandle.getHandle();

      auto result = evalCollectivePermuteOp(operand, sourceTargetPairs,
                                            channelId, process);
      scope.add(collectivePermuteOp.getResult(), result);
    } else if (auto compareOp = dyn_cast<CompareOp>(op)) {
      auto lhs = scope.findTensor(compareOp.getLhs());
      auto rhs = scope.findTensor(compareOp.getRhs());
      auto comparisonDirection = compareOp.getComparisonDirection();
      auto result =
          evalCompareOp(lhs, rhs, comparisonDirection, compareOp.getType());
      scope.add(compareOp.getResult(), result);
    } else if (auto complexOp = dyn_cast<ComplexOp>(op)) {
      auto lhs = scope.findTensor(complexOp.getLhs());
      auto rhs = scope.findTensor(complexOp.getRhs());
      auto result = evalComplexOp(lhs, rhs, complexOp.getType());
      scope.add(complexOp.getResult(), result);
    } else if (auto concatenateOp = dyn_cast<ConcatenateOp>(op)) {
      auto operands = scope.findTensors(concatenateOp.getOperands());
      auto result = evalConcatenateOp(operands, concatenateOp.getDimension(),
                                      concatenateOp.getType());
      scope.add(concatenateOp.getResult(), result);
    } else if (auto constantOp = dyn_cast<ConstantOp>(op)) {
      auto result = evalConstantOp(constantOp.getValue());
      scope.add(constantOp.getResult(), result);
    } else if (auto convertOp = dyn_cast<ConvertOp>(op)) {
      auto operand = scope.findTensor(convertOp.getOperand());
      auto result = evalConvertOp(operand, convertOp.getType());
      scope.add(convertOp.getResult(), result);
    } else if (auto cosineOp = dyn_cast<CosineOp>(op)) {
      auto operand = scope.findTensor(cosineOp.getOperand());
      auto result = evalCosineOp(operand, cosineOp.getType());
      scope.add(cosineOp.getResult(), result);
    } else if (isa<CreateTokenOp>(op)) {
      failOnDecomposableOp(op);
    } else if (isa<CrossReplicaSumOp>(op)) {
      failOnDecomposableOp(op);
    } else if (auto divideOp = dyn_cast<DivOp>(op)) {
      auto lhs = scope.findTensor(divideOp.getLhs());
      auto rhs = scope.findTensor(divideOp.getRhs());
      auto result = evalDivideOp(lhs, rhs, divideOp.getType());
      scope.add(divideOp.getResult(), result);
    } else if (isa<DotOp>(op)) {
      failOnDecomposableOp(op);
    } else if (auto dotGeneralOp = dyn_cast<DotGeneralOp>(op)) {
      auto lhs = scope.findTensor(dotGeneralOp.getLhs());
      auto rhs = scope.findTensor(dotGeneralOp.getRhs());
      auto lhsBatchingDimensions = Axes(
          dotGeneralOp.getDotDimensionNumbers().getLhsBatchingDimensions());
      auto rhsBatchingDimensions = Axes(
          dotGeneralOp.getDotDimensionNumbers().getRhsBatchingDimensions());
      auto lhsContractingDimensions = Axes(
          dotGeneralOp.getDotDimensionNumbers().getLhsContractingDimensions());
      auto rhsContractingDimensions = Axes(
          dotGeneralOp.getDotDimensionNumbers().getRhsContractingDimensions());
      auto result =
          evalDotGeneralOp(lhs, rhs, lhsBatchingDimensions,
                           rhsBatchingDimensions, lhsContractingDimensions,
                           rhsContractingDimensions, dotGeneralOp.getType());
      scope.add(dotGeneralOp.getResult(), result);
    } else if (auto dynamicSliceOp = dyn_cast<DynamicSliceOp>(op)) {
      auto operand = scope.findTensor(dynamicSliceOp.getOperand());
      auto startIndices = scope.findTensors(dynamicSliceOp.getStartIndices());
      auto sliceSizes = Sizes(dynamicSliceOp.getSliceSizes());
      auto result = evalDynamicSliceOp(operand, startIndices, sliceSizes,
                                       dynamicSliceOp.getType());
      scope.add(dynamicSliceOp.getResult(), result);
    } else if (auto dynamicUpdateSliceOp = dyn_cast<DynamicUpdateSliceOp>(op)) {
      auto operand = scope.findTensor(dynamicUpdateSliceOp.getOperand());
      auto update = scope.findTensor(dynamicUpdateSliceOp.getUpdate());
      auto startIndices =
          scope.findTensors(dynamicUpdateSliceOp.getStartIndices());
      auto result = evalDynamicUpdateSliceOp(operand, update, startIndices,
                                             dynamicUpdateSliceOp.getType());
      scope.add(dynamicUpdateSliceOp.getResult(), result);
    } else if (isa<EinsumOp>(op)) {
      failOnDecomposableOp(op);
    } else if (auto expOp = dyn_cast<ExpOp>(op)) {
      auto operand = scope.findTensor(expOp.getOperand());
      auto result = evalExponentialOp(operand, expOp.getType());
      scope.add(expOp.getResult(), result);
    } else if (auto expm1Op = dyn_cast<Expm1Op>(op)) {
      auto operand = scope.findTensor(expm1Op.getOperand());
      auto result = evalExpm1Op(operand, expm1Op.getType());
      scope.add(expm1Op.getResult(), result);
    } else if (auto floorOp = dyn_cast<FloorOp>(op)) {
      auto operand = scope.findTensor(floorOp.getOperand());
      auto result = evalFloorOp(operand, floorOp.getType());
      scope.add(floorOp.getResult(), result);
    } else if (auto gatherOp = dyn_cast<GatherOp>(op)) {
      auto operand = scope.findTensor(gatherOp.getOperand());
      auto startIndices = scope.findTensor(gatherOp.getStartIndices());
      auto result = evalGatherOp(
          operand, startIndices,
          Axes(gatherOp.getDimensionNumbers().getOffsetDims()),
          Axes(gatherOp.getDimensionNumbers().getCollapsedSliceDims()),
          Axes(gatherOp.getDimensionNumbers().getStartIndexMap()),
          Axis(gatherOp.getDimensionNumbers().getIndexVectorDim()),
          Sizes(gatherOp.getSliceSizes()), gatherOp.getIndicesAreSorted(),
          gatherOp.getType());
      scope.add(gatherOp.getResult(), result);
    } else if (auto getDimensionSizeOp = dyn_cast<GetDimensionSizeOp>(op)) {
      auto operand = scope.findTensor(getDimensionSizeOp.getOperand());
      auto dimension = getDimensionSizeOp.getDimension();
      auto result = evalGetDimensionSizeOp(operand, dimension,
                                           getDimensionSizeOp.getType());
      scope.add(getDimensionSizeOp.getResult(), result);
    } else if (auto ifOp = dyn_cast<IfOp>(op)) {
      auto pred = scope.findTensor(ifOp.getPred());
      auto &trueBranch = ifOp.getTrueBranch();
      auto &falseBranch = ifOp.getFalseBranch();
      auto results = evalIfOp(pred, trueBranch, falseBranch, process, scope);
      scope.add(ifOp.getResults(), results);
    } else if (auto imagOp = dyn_cast<ImagOp>(op)) {
      auto operand = scope.findTensor(imagOp.getOperand());
      auto result = evalImagOp(operand, imagOp.getType());
      scope.add(imagOp.getResult(), result);
    } else if (auto iotaOp = dyn_cast<IotaOp>(op)) {
      auto iotaDimension = iotaOp.getIotaDimension();
      auto result = evalIotaOp(iotaDimension, iotaOp.getType());
      scope.add(iotaOp.getResult(), result);
    } else if (auto isFiniteOp = dyn_cast<IsFiniteOp>(op)) {
      auto operand = scope.findTensor(isFiniteOp.getOperand());
      auto result = evalIsFiniteOp(operand, isFiniteOp.getType());
      scope.add(isFiniteOp.getResult(), result);
    } else if (auto log1pOp = dyn_cast<Log1pOp>(op)) {
      auto operand = scope.findTensor(log1pOp.getOperand());
      auto result = evalLog1pOp(operand, log1pOp.getType());
      scope.add(log1pOp.getResult(), result);
    } else if (auto logOp = dyn_cast<LogOp>(op)) {
      auto operand = scope.findTensor(logOp.getOperand());
      auto result = evalLogOp(operand, logOp.getType());
      scope.add(logOp.getResult(), result);
    } else if (auto logisticOp = dyn_cast<LogisticOp>(op)) {
      auto operand = scope.findTensor(logisticOp.getOperand());
      auto result = evalLogisticOp(operand, logisticOp.getType());
      scope.add(logisticOp.getResult(), result);
    } else if (auto mapOp = dyn_cast<MapOp>(op)) {
      auto inputs = scope.findTensors(mapOp.getInputs());
      auto &computation = mapOp.getComputation();
      auto result =
          evalMapOp(inputs, computation, process, scope, mapOp.getType());
      scope.add(mapOp.getResult(), result);
    } else if (auto maxOp = dyn_cast<MaxOp>(op)) {
      auto lhs = scope.findTensor(maxOp.getLhs());
      auto rhs = scope.findTensor(maxOp.getRhs());
      auto result = evalMaxOp(lhs, rhs, maxOp.getType());
      scope.add(maxOp.getResult(), result);
    } else if (auto minOp = dyn_cast<MinOp>(op)) {
      auto lhs = scope.findTensor(minOp.getLhs());
      auto rhs = scope.findTensor(minOp.getRhs());
      auto result = evalMinOp(lhs, rhs, minOp.getType());
      scope.add(minOp.getResult(), result);
    } else if (auto multiplyOp = dyn_cast<MulOp>(op)) {
      auto lhs = scope.findTensor(multiplyOp.getLhs());
      auto rhs = scope.findTensor(multiplyOp.getRhs());
      auto result = evalMultiplyOp(lhs, rhs, multiplyOp.getType());
      scope.add(multiplyOp.getResult(), result);
    } else if (auto negOp = dyn_cast<NegOp>(op)) {
      auto operand = scope.findTensor(negOp.getOperand());
      auto result = evalNegOp(operand, negOp.getType());
      scope.add(negOp.getResult(), result);
    } else if (auto notOp = dyn_cast<NotOp>(op)) {
      auto operand = scope.findTensor(notOp.getOperand());
      auto result = evalNotOp(operand, notOp.getType());
      scope.add(notOp.getResult(), result);
    } else if (auto optimizationBarrierOp =
                   dyn_cast<OptimizationBarrierOp>(op)) {
      auto operand = scope.find(optimizationBarrierOp.getOperand());
      auto results = evalOptimizationBarrierOp(operand);
      scope.add(optimizationBarrierOp.getResults(), results);
    } else if (auto orOp = dyn_cast<OrOp>(op)) {
      auto lhs = scope.findTensor(orOp.getLhs());
      auto rhs = scope.findTensor(orOp.getRhs());
      auto result = evalOrOp(lhs, rhs, orOp.getType());
      scope.add(orOp.getResult(), result);
    } else if (auto outfeedOp = dyn_cast<OutfeedOp>(op)) {
      auto inputs = scope.findTensors(outfeedOp.getInputs());
      auto token = scope.findToken(outfeedOp.getToken());
      auto result = evalOutfeedOp(inputs, token, process);
      scope.add(outfeedOp.getResult(), result);
    } else if (auto padOp = dyn_cast<PadOp>(op)) {
      auto operand = scope.findTensor(padOp.getOperand());
      auto paddingValue = scope.findTensor(padOp.getPaddingValue());
      auto edgePaddingLow = Sizes(padOp.getEdgePaddingLow());
      auto interiorPadding = Sizes(padOp.getInteriorPadding());
      auto result = evalPadOp(operand, paddingValue, edgePaddingLow,
                              interiorPadding, padOp.getType());
      scope.add(padOp.getResult(), result);
    } else if (auto partitionIdOp = dyn_cast<PartitionIdOp>(op)) {
      auto result = evalPartitionIdOp(process, op.getContext());
      scope.add(partitionIdOp.getResult(), result);
    } else if (auto populationCountOp = dyn_cast<PopulationCountOp>(op)) {
      auto operand = scope.findTensor(populationCountOp.getOperand());
      auto result = evalPopulationCountOp(operand, populationCountOp.getType());
      scope.add(populationCountOp.getResult(), result);
    } else if (auto powerOp = dyn_cast<PowOp>(op)) {
      auto lhs = scope.findTensor(powerOp.getLhs());
      auto rhs = scope.findTensor(powerOp.getRhs());
      auto result = evalPowerOp(lhs, rhs, powerOp.getType());
      scope.add(powerOp.getResult(), result);
    } else if (auto realOp = dyn_cast<RealOp>(op)) {
      auto operand = scope.findTensor(realOp.getOperand());
      auto result = evalRealOp(operand, realOp.getType());
      scope.add(realOp.getResult(), result);
    } else if (auto reduceOp = dyn_cast<ReduceOp>(op)) {
      auto inputs = scope.findTensors(reduceOp.getInputs());
      auto initValues = scope.findTensors(reduceOp.getInitValues());
      SmallVector<ShapedType> resultTypes;
      for (auto resultType : reduceOp.getResultTypes())
        resultTypes.push_back(resultType.cast<ShapedType>());
      auto results =
          evalReduceOp(inputs, initValues, Axes(reduceOp.getDimensions()),
                       reduceOp.getBody(), process, scope, resultTypes);
      scope.add(reduceOp.getResults(), results);
    } else if (auto reducePrecisionOp = dyn_cast<ReducePrecisionOp>(op)) {
      auto operand = scope.findTensor(reducePrecisionOp.getOperand());
      int32_t exponentBits = reducePrecisionOp.getExponentBits();
      int32_t mantissaBits = reducePrecisionOp.getMantissaBits();
      auto result = evalReducePrecisionOp(operand, exponentBits, mantissaBits,
                                          reducePrecisionOp.getType());
      scope.add(reducePrecisionOp.getResult(), result);
    } else if (auto reduceWindowOp = dyn_cast<ReduceWindowOp>(op)) {
      auto inputs = scope.findTensors(reduceWindowOp.getInputs());
      auto initValues = scope.findTensors(reduceWindowOp.getInitValues());
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

      auto results = evalReduceWindowOp(
          inputs, initValues, Sizes(reduceWindowOp.getWindowDimensions()),
          windowStrides, baseDilations, windowDilations, paddingLow,
          paddingHigh, reduceWindowOp.getBody(), process, scope, resultTypes);
      scope.add(reduceWindowOp.getResults(), results);
    } else if (auto remOp = dyn_cast<RemOp>(op)) {
      auto lhs = scope.findTensor(remOp.getLhs());
      auto rhs = scope.findTensor(remOp.getRhs());
      auto result = evalRemOp(lhs, rhs, remOp.getType());
      scope.add(remOp.getResult(), result);
    } else if (auto replicaIdOp = dyn_cast<ReplicaIdOp>(op)) {
      auto result = evalReplicaIdOp(process, op.getContext());
      scope.add(replicaIdOp.getResult(), result);
    } else if (auto reshapeOp = dyn_cast<ReshapeOp>(op)) {
      auto operand = scope.findTensor(reshapeOp.getOperand());
      auto result = evalReshapeOp(operand, reshapeOp.getType());
      scope.add(reshapeOp.getResult(), result);
    } else if (auto returnOp = dyn_cast<func::ReturnOp>(op)) {
      return scope.find(returnOp.getOperands());
    } else if (auto returnOp = dyn_cast<ReturnOp>(op)) {
      return scope.find(returnOp.getResults());
    } else if (auto reverseOp = dyn_cast<ReverseOp>(op)) {
      auto operand = scope.findTensor(reverseOp.getOperand());
      auto dimensions = Axes(reverseOp.getDimensions());
      auto result = evalReverseOp(operand, dimensions, reverseOp.getType());
      scope.add(reverseOp.getResult(), result);
    } else if (isa<RngBitGeneratorOp>(op)) {
      failOnDecomposableOp(op);
    } else if (isa<RngOp>(op)) {
      failOnDecomposableOp(op);
    } else if (auto roundNearestEvenOp = dyn_cast<RoundNearestEvenOp>(op)) {
      auto operand = scope.findTensor(roundNearestEvenOp.getOperand());
      auto result =
          evalRoundNearestEvenOp(operand, roundNearestEvenOp.getType());
      scope.add(roundNearestEvenOp.getResult(), result);
    } else if (auto roundOp = dyn_cast<RoundOp>(op)) {
      auto operand = scope.findTensor(roundOp.getOperand());
      auto result = evalRoundOp(operand, roundOp.getType());
      scope.add(roundOp.getResult(), result);
    } else if (auto rsqrtOp = dyn_cast<RsqrtOp>(op)) {
      auto operand = scope.findTensor(rsqrtOp.getOperand());
      auto result = evalRsqrtOp(operand, rsqrtOp.getType());
      scope.add(rsqrtOp.getResult(), result);
    } else if (auto scatterOp = dyn_cast<ScatterOp>(op)) {
      auto inputs = scope.findTensors(scatterOp.getInputs());
      auto scatterIndices = scope.findTensor(scatterOp.getScatterIndices());
      auto updates = scope.findTensors(scatterOp.getUpdates());
      auto scatterDimensionNumbers = scatterOp.getScatterDimensionNumbersAttr();
      Axes updateWindowDims(scatterDimensionNumbers.getUpdateWindowDims());
      Axes insertedWindowDims(scatterDimensionNumbers.getInsertedWindowDims());
      Axes scatterDimsToOperandDims(
          scatterDimensionNumbers.getScatterDimsToOperandDims());
      Axis indexVectorDim(scatterDimensionNumbers.getIndexVectorDim());
      auto &updateComputation = scatterOp.getUpdateComputation();
      SmallVector<ShapedType> resultTypes(scatterOp->getResultTypes());
      auto results = evalScatterOp(
          inputs, scatterIndices, updates, updateWindowDims, insertedWindowDims,
          scatterDimsToOperandDims, indexVectorDim, updateComputation, process,
          scope, resultTypes);
      scope.add(scatterOp.getResults(), results);
    } else if (auto selectAndScatterOp = dyn_cast<SelectAndScatterOp>(op)) {
      auto operand = scope.findTensor(selectAndScatterOp.getOperand());
      auto source = scope.findTensor(selectAndScatterOp.getSource());
      auto initValue = scope.findTensor(selectAndScatterOp.getInitValue());
      auto rank = operand.getRank();

      Sizes windowDimensions(rank, 1);
      if (auto windowDimensionsAttr =
              selectAndScatterOp.getWindowDimensionsAttr())
        windowDimensions.assign(
            windowDimensionsAttr.getValues<int64_t>().begin(),
            windowDimensionsAttr.getValues<int64_t>().end());

      Sizes windowStrides(rank, 1);
      if (auto windowStridesAttr = selectAndScatterOp.getWindowStridesAttr())
        windowStrides.assign(windowStridesAttr.getValues<int64_t>().begin(),
                             windowStridesAttr.getValues<int64_t>().end());

      Sizes paddingLow(rank, 0);
      if (auto padding = selectAndScatterOp.getPadding()) {
        auto paddingOrErr = hlo::convertPaddingAttribute(padding, {});
        if (failed(paddingOrErr))
          report_fatal_error(invalidArgument("Invalid padding format found."));
        for (auto i = 0; i < static_cast<int64_t>(paddingOrErr->size()); ++i) {
          paddingLow[i] = (*paddingOrErr)[i].first;
        }
      }

      auto result = evalSelectAndScatterOp(
          operand, source, initValue, windowDimensions, windowStrides,
          paddingLow, selectAndScatterOp.getSelect(),
          selectAndScatterOp.getScatter(), process, scope,
          selectAndScatterOp.getType());
      scope.add(selectAndScatterOp.getResult(), result);
    } else if (auto selectOp = dyn_cast<SelectOp>(op)) {
      auto pred = scope.findTensor(selectOp.getPred());
      auto onTrue = scope.findTensor(selectOp.getOnTrue());
      auto onFalse = scope.findTensor(selectOp.getOnFalse());
      auto result = evalSelectOp(pred, onTrue, onFalse, selectOp.getType());
      scope.add(selectOp.getResult(), result);
    } else if (auto shiftLeftOp = dyn_cast<ShiftLeftOp>(op)) {
      auto lhs = scope.findTensor(shiftLeftOp.getLhs());
      auto rhs = scope.findTensor(shiftLeftOp.getRhs());
      auto result = evalShiftLeftOp(lhs, rhs, shiftLeftOp.getType());
      scope.add(shiftLeftOp.getResult(), result);
    } else if (auto shiftRightArithmeticOp =
                   dyn_cast<ShiftRightArithmeticOp>(op)) {
      auto lhs = scope.findTensor(shiftRightArithmeticOp.getLhs());
      auto rhs = scope.findTensor(shiftRightArithmeticOp.getRhs());
      auto result = evalShiftRightArithmeticOp(
          lhs, rhs, shiftRightArithmeticOp.getType());
      scope.add(shiftRightArithmeticOp.getResult(), result);
    } else if (auto shiftRightLogicalOp = dyn_cast<ShiftRightLogicalOp>(op)) {
      auto lhs = scope.findTensor(shiftRightLogicalOp.getLhs());
      auto rhs = scope.findTensor(shiftRightLogicalOp.getRhs());
      auto result =
          evalShiftRightLogicalOp(lhs, rhs, shiftRightLogicalOp.getType());
      scope.add(shiftRightLogicalOp.getResult(), result);
    } else if (auto signOp = dyn_cast<SignOp>(op)) {
      auto operand = scope.findTensor(signOp.getOperand());
      auto result = evalSignOp(operand, signOp.getType());
      scope.add(signOp.getResult(), result);
    } else if (auto sineOp = dyn_cast<SineOp>(op)) {
      auto operand = scope.findTensor(sineOp.getOperand());
      auto result = evalSineOp(operand, sineOp.getType());
      scope.add(sineOp.getResult(), result);
    } else if (auto sliceOp = dyn_cast<SliceOp>(op)) {
      auto operand = scope.findTensor(sliceOp.getOperand());
      auto startIndices = Sizes(sliceOp.getStartIndices());
      auto strides = Sizes(sliceOp.getStrides());
      auto result =
          evalSliceOp(operand, startIndices, strides, sliceOp.getType());
      scope.add(sliceOp.getResult(), result);
    } else if (auto sortOp = dyn_cast<SortOp>(op)) {
      auto operands = scope.findTensors(sortOp.getInputs());
      auto dimension = sortOp.getDimension();
      auto isStable = sortOp.getIsStable();
      auto &comparator = sortOp.getComparator();
      auto results =
          evalSortOp(operands, dimension, isStable, comparator, process, scope);
      scope.add(sortOp.getResults(), results);
    } else if (auto sqrtOp = dyn_cast<SqrtOp>(op)) {
      auto operand = scope.findTensor(sqrtOp.getOperand());
      auto result = evalSqrtOp(operand, sqrtOp.getType());
      scope.add(sqrtOp.getResult(), result);
    } else if (auto subtractOp = dyn_cast<SubtractOp>(op)) {
      auto lhs = scope.findTensor(subtractOp.getLhs());
      auto rhs = scope.findTensor(subtractOp.getRhs());
      auto result = evalSubtractOp(lhs, rhs, subtractOp.getType());
      scope.add(subtractOp.getResult(), result);
    } else if (auto tanhOp = dyn_cast<TanhOp>(op)) {
      auto operand = scope.findTensor(tanhOp.getOperand());
      auto result = evalTanhOp(operand, tanhOp.getType());
      scope.add(tanhOp.getResult(), result);
    } else if (isa<TorchIndexSelectOp>(op)) {
      failOnDecomposableOp(op);
    } else if (isa<TraceOp>(op)) {
      failOnDecomposableOp(op);
    } else if (auto transposeOp = dyn_cast<TransposeOp>(op)) {
      auto operand = scope.findTensor(transposeOp.getOperand());
      auto permutation = Axes(transposeOp.getPermutation());
      auto result =
          evalTransposeOp(operand, permutation, transposeOp.getType());
      scope.add(transposeOp.getResult(), result);
    } else if (isa<TriangularSolveOp>(op)) {
      failOnDecomposableOp(op);
    } else if (isa<UnaryEinsumOp>(op)) {
      failOnDecomposableOp(op);
    } else if (auto whileOp = dyn_cast<WhileOp>(op)) {
      auto operand = scope.find(whileOp.getOperand());
      auto &cond = whileOp.getCond();
      auto &body = whileOp.getBody();
      auto results = evalWhileOp(operand, cond, body, process, scope);
      scope.add(whileOp.getResults(), results);
    } else if (auto xorOp = dyn_cast<XorOp>(op)) {
      auto lhs = scope.findTensor(xorOp.getLhs());
      auto rhs = scope.findTensor(xorOp.getRhs());
      auto result = evalXorOp(lhs, rhs, xorOp.getType());
      scope.add(xorOp.getResult(), result);
    } else {
      if (!fallback)
        report_fatal_error(
            invalidArgument("Unsupported op: %s", debugString(op).c_str()));
      auto status = fallback(op, process, scope);
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

Token evalAfterAllOp(ArrayRef<Token> inputs, MLIRContext *context) {
  return Token(context);
}

Tensor evalAllReduceOp(const Tensor &operand,
                       SmallVector<SmallVector<uint32_t>> replicaGroups,
                       int64_t channelId, bool useGlobalDeviceIds,
                       Region &computation, Process *process, Scope &scope,
                       ShapedType resultType) {
  if (!process)
    llvm::report_fatal_error(
        "all_reduce is only supported when run via interpreter.run_parallel");

  ProcessGroups processGroups;
  if (channelId <= 0 && !useGlobalDeviceIds)
    processGroups = process->crossReplica(replicaGroups);
  if (channelId > 0 && !useGlobalDeviceIds)
    processGroups = process->crossReplicaAndPartition(replicaGroups);
  if (channelId > 0 && useGlobalDeviceIds)
    processGroups = process->flattenedIds(replicaGroups);

  auto processGroup = processGroups.findGroup(process->getId());
  if (!processGroup)
    llvm::report_fatal_error(invalidArgument(
        "Failed to find process group with process_id: (%d, %d)",
        process->getId().replicaId, process->getId().partitionId));

  auto groupOperands =
      process->rendezvous(*processGroup, channelId, operand).getSortedTensors();

  Tensor result(resultType);
  for (auto resultIt = result.index_begin(); resultIt != result.index_end();
       ++resultIt) {
    Tensor resultElement;
    for (auto groupOperand : groupOperands) {
      auto groupOperandElement = makeScalar(groupOperand.get(*resultIt));
      if (resultElement)
        resultElement = eval(computation, {resultElement, groupOperandElement},
                             process, &scope)[0]
                            .getTensor();
      else
        resultElement = groupOperandElement;
    }
    result.set(*resultIt, resultElement.get({}));
  }

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

Tensor evalBitcastConvertOp(const Tensor &operand, ShapedType resultType) {
  Tensor result(resultType);

  auto resultElementType = result.getElementType();
  auto resultNumBits = numBits(result.getElementType());
  auto operandNumBits = numBits(operand.getElementType());

  if (resultNumBits < operandNumBits) {
    auto resultIt = result.index_begin();
    for (auto operandIt = operand.index_begin();
         operandIt != operand.index_end(); ++operandIt) {
      auto resultElements =
          bitcastConvertOneToMany(resultElementType, operand.get(*operandIt));
      for (const auto &resultElement : resultElements)
        result.set(*resultIt++, resultElement);
    }
    return result;
  }

  if (resultNumBits > operandNumBits) {
    auto operandIt = operand.index_begin();
    for (auto resultIt = result.index_begin(); resultIt != result.index_end();
         ++resultIt) {
      SmallVector<Element> operandElements;
      for (auto i = 0; i < resultNumBits / operandNumBits; ++i)
        operandElements.push_back(operand.get(*operandIt++));
      result.set(*resultIt,
                 bitcastConvertManyToOne(resultElementType, operandElements));
    }
    return result;
  }

  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it,
               bitcastConvertOneToOne(resultElementType, operand.get(*it)));
  return result;
}

Tensor evalBroadcastInDimOp(const Tensor &operand,
                            const Axes &broadcastDimensions,
                            ShapedType resultType) {
  Tensor result(resultType);
  for (auto resultIt = result.index_begin(); resultIt != result.index_end();
       ++resultIt) {
    auto resultIndex = *resultIt;
    Index operandIndex(operand.getRank(), 0);
    for (auto d = 0; d < operand.getRank(); ++d) {
      if (operand.getShape()[d] == 1) continue;
      operandIndex[d] = resultIndex[broadcastDimensions[d]];
    }
    result.set(resultIndex, operand.get(operandIndex));
  }
  return result;
}

SmallVector<InterpreterValue> evalCaseOp(const Tensor &index,
                                         RegionRange branches, Process *process,
                                         Scope &scope) {
  int64_t indexValue = index.get({}).getIntegerValue().getSExtValue();
  if (indexValue < 0 || indexValue >= static_cast<int64_t>(branches.size()))
    indexValue = branches.size() - 1;

  return eval(*branches[indexValue], {}, process, &scope);
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

Tensor evalCollectivePermuteOp(
    const Tensor &operand, SmallVector<SmallVector<uint32_t>> sourceTargetPairs,
    ChannelId channelId, Process *process) {
  if (!process)
    llvm::report_fatal_error(
        "collective_permute is only supported when run via "
        "interpreter.run_parallel");

  ProcessGroups processGroups;
  if (channelId <= 0) processGroups = process->crossReplica(sourceTargetPairs);
  if (channelId > 0) processGroups = process->crossPartition(sourceTargetPairs);

  Tensor result;
  for (auto processGroup : processGroups) {
    auto from = processGroup[0];
    auto to = processGroup[1];
    if (from != process->getId() && to != process->getId()) continue;

    auto rendezvousResult =
        process->rendezvous(processGroup, channelId, operand);
    if (to != process->getId()) continue;
    result = rendezvousResult.lookup(from);
  }

  if (result) return result;
  return evalBroadcastInDimOp(
      makeScalar(convert(operand.getElementType(), 0.0)), {},
      operand.getType());
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
      auto inputIndex = *inputIt;
      Index resultIndex(inputIndex);
      resultIndex[dimension] += dimensionOffset;
      result.set(resultIndex, input.get(inputIndex));
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

Tensor evalDivideOp(const Tensor &lhs, const Tensor &rhs,
                    ShapedType resultType) {
  Tensor result(resultType);
  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it, lhs.get(*it) / rhs.get(*it));
  return result;
}

Tensor evalDotGeneralOp(const Tensor &lhs, const Tensor &rhs,
                        const Axes &lhsBatchingDimensions,
                        const Axes &rhsBatchingDimensions,
                        const Axes &lhsContractingDimensions,
                        const Axes &rhsContractingDimensions,
                        ShapedType resultType) {
  Tensor result(resultType);
  Axes lhsResultDims;
  for (auto i = 0; i < lhs.getType().getRank(); ++i)
    if (!llvm::is_contained(lhsBatchingDimensions, i) &&
        !llvm::is_contained(lhsContractingDimensions, i))
      lhsResultDims.push_back(i);

  Axes rhsResultDims;
  for (auto i = 0; i < rhs.getType().getRank(); ++i)
    if (!llvm::is_contained(rhsBatchingDimensions, i) &&
        !llvm::is_contained(rhsContractingDimensions, i))
      rhsResultDims.push_back(i);

  for (auto resultIt = result.index_begin(); resultIt != result.index_end();
       ++resultIt) {
    // Each result element is computed as dot product of slices of lhs and rhs.
    // In this implementation, we aren't going to materialize these slices as
    // standalone tensors, but are going to iterate through lhs and rhs
    // via lhsIndex and rhsIndex.
    auto resultIndex = *resultIt;
    Index lhsIndex(lhs.getType().getRank(), 0);
    Index rhsIndex(rhs.getType().getRank(), 0);

    // Some pieces of lhsIndex and rhsIndex stay the same during iteration.
    // These are the indices that correspond to non-contracting dimensions,
    // and they are initialized here.
    int64_t resultDim = 0;
    for (size_t i = 0; i < lhsBatchingDimensions.size(); ++i, ++resultDim) {
      lhsIndex[lhsBatchingDimensions[i]] = resultIndex[resultDim];
      rhsIndex[rhsBatchingDimensions[i]] = resultIndex[resultDim];
    }
    for (size_t i = 0; i < lhsResultDims.size(); ++i, ++resultDim)
      lhsIndex[lhsResultDims[i]] = resultIndex[resultDim];
    for (size_t i = 0; i < rhsResultDims.size(); ++i, ++resultDim)
      rhsIndex[rhsResultDims[i]] = resultIndex[resultDim];

    // Iteration space is defined by contracting dimensions.
    // The corresponding parts of lhsIndex and rhsIndex start at 0, 0, ..., 0.
    // Then, we increment them lexicographically until we're out of bounds.
    auto incrementIndices = [&]() -> LogicalResult {
      // Implementation is heavily inspired by IndexSpaceIterator::operator++.
      if (lhsContractingDimensions.empty()) return failure();
      for (int64_t i = lhsContractingDimensions.size() - 1; i >= 0; --i) {
        lhsIndex[lhsContractingDimensions[i]]++;
        rhsIndex[rhsContractingDimensions[i]]++;
        if (lhsIndex[lhsContractingDimensions[i]] <
            lhs.getShape()[lhsContractingDimensions[i]])
          return success();
        if (i == 0) return failure();
        lhsIndex[lhsContractingDimensions[i]] = 0;
        rhsIndex[rhsContractingDimensions[i]] = 0;
      }
      return success();
    };

    // Now that the lhsIndex/rhsIndex and the iteration space are set up,
    // we can compute the dot product of the (virtual) slices of lhs and rhs.
    auto resultElement = convert(resultType.getElementType(), 0.0);
    while (true) {
      resultElement = resultElement + lhs.get(lhsIndex) * rhs.get(rhsIndex);
      if (failed(incrementIndices())) break;
    }
    result.set(resultIndex, resultElement);
  }
  return result;
}

Tensor evalDynamicSliceOp(const Tensor &operand, ArrayRef<Tensor> startIndices,
                          const Sizes &sliceSizes, ShapedType resultType) {
  Tensor result(resultType);
  auto adjustedStartIndices =
      clamp(0, evalIndex(startIndices), operand.getShape() - sliceSizes);
  for (auto resultIt = result.index_begin(); resultIt != result.index_end();
       ++resultIt) {
    auto resultIndex = *resultIt;
    auto operandIndex = adjustedStartIndices + *resultIt;
    result.set(resultIndex, operand.get(operandIndex));
  }
  return result;
}

Tensor evalDynamicUpdateSliceOp(const Tensor &operand, const Tensor &update,
                                ArrayRef<Tensor> startIndices,
                                ShapedType resultType) {
  Tensor result(resultType);
  auto adjustedStartIndices =
      clamp(0, evalIndex(startIndices), operand.getShape() - update.getShape());
  for (auto resultIt = result.index_begin(); resultIt != result.index_end();
       ++resultIt) {
    auto resultIndex = *resultIt;
    auto updateIndex = resultIndex - adjustedStartIndices;
    if (updateIndex.inBounds(update.getShape()))
      result.set(resultIndex, update.get(updateIndex));
    else
      result.set(resultIndex, operand.get(resultIndex));
  }
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

Tensor evalGatherOp(const Tensor &operand, const Tensor &startIndices,
                    const Axes &offsetDims, const Axes &collapsedSliceDims,
                    const Axes &startIndexMap, Axis indexVectorDim,
                    const Sizes &sliceSizes, bool indicesAreSorted,
                    ShapedType resultType) {
  Tensor result(resultType);
  Axes batchDims;
  for (auto d : result.getAxes())
    if (!llvm::is_contained(offsetDims, d)) batchDims.push_back(d);

  for (auto resultIt = result.index_begin(); resultIt != result.index_end();
       ++resultIt) {
    auto resultIndex = *resultIt;

    Index batchIndex;
    for (auto d : batchDims) batchIndex.push_back(resultIndex[d]);

    auto startIndicesIndex = batchIndex;
    if (indexVectorDim < startIndices.getRank())
      startIndicesIndex.insert(startIndicesIndex.begin() + indexVectorDim,
                               kColon);
    auto startIndex = evalIndex(evalSliceOp(startIndices, startIndicesIndex));

    Index fullStartIndex(operand.getRank(), 0);
    for (auto dOperand : operand.getAxes()) {
      auto dStartIt = llvm::find(startIndexMap, dOperand);
      if (dStartIt == startIndexMap.end()) continue;
      auto dStart = dStartIt - startIndexMap.begin();
      fullStartIndex[dOperand] = std::clamp<int64_t>(
          startIndex[dStart], 0L,
          operand.getShape()[dOperand] - sliceSizes[dOperand]);
    }

    Index offsetIndex;
    for (auto d : offsetDims) offsetIndex.push_back(resultIndex[d]);

    Index fullOffsetIndex(offsetIndex.size() + collapsedSliceDims.size(), 0);
    for (size_t i = 0, oi = 0; i < fullOffsetIndex.size(); ++i) {
      if (llvm::is_contained(collapsedSliceDims, i)) continue;
      fullOffsetIndex[i] = offsetIndex[oi++];
    }

    auto operandIndex = fullStartIndex + fullOffsetIndex;
    result.set(resultIndex, operand.get(operandIndex));
  }
  return result;
}

Tensor evalGetDimensionSizeOp(const Tensor &operand, Axis dimension,
                              ShapedType resultType) {
  Tensor result(resultType);
  result.set(
      {}, convert(resultType.getElementType(), operand.getShape()[dimension]));
  return result;
}

SmallVector<InterpreterValue> evalIfOp(const Tensor &pred, Region &trueBranch,
                                       Region &falseBranch, Process *process,
                                       Scope &scope) {
  return pred.get({}).getBooleanValue()
             ? eval(trueBranch, {}, process, &scope)
             : eval(falseBranch, {}, process, &scope);
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

Tensor evalMapOp(ArrayRef<Tensor> inputs, Region &computation, Process *process,
                 Scope &scope, ShapedType resultType) {
  Tensor result(resultType);
  for (auto it = result.index_begin(); it != result.index_end(); ++it) {
    SmallVector<InterpreterValue> args;
    for (size_t i = 0; i < inputs.size(); ++i) {
      Tensor tensor(computation.getArgument(i).getType().cast<ShapedType>());
      tensor.set({}, inputs[i].get(*it));
      args.emplace_back(tensor);
    }
    result.set(*it,
               eval(computation, args, process, &scope)[0].getTensor().get({}));
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

SmallVector<InterpreterValue> evalOptimizationBarrierOp(
    ArrayRef<InterpreterValue> operand) {
  return SmallVector<InterpreterValue>(operand);
}

Tensor evalOrOp(const Tensor &lhs, const Tensor &rhs, ShapedType resultType) {
  Tensor result(resultType);
  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it, lhs.get(*it) | rhs.get(*it));
  return result;
}

Token evalOutfeedOp(ArrayRef<Tensor> inputs, Token token, Process *process) {
  if (!process)
    llvm::report_fatal_error(
        "outfeed is only supported when run via interpreter.run_parallel");

  process->outfeed(inputs);
  return token;
}

Tensor evalPadOp(const Tensor &operand, const Tensor &paddingValue,
                 const Sizes &edgePaddingLow, const Sizes &interiorPadding,
                 ShapedType resultType) {
  auto result = makeSplat(resultType, paddingValue.get({}));
  for (auto operandIt = operand.index_begin(); operandIt != operand.index_end();
       ++operandIt) {
    auto operandIndex = *operandIt;
    auto resultIndex = edgePaddingLow + operandIndex * (interiorPadding + 1);
    // Bound check is needed here because of negative padding which could
    // swallow some operand indices.
    if (resultIndex.inBounds(result.getShape()))
      result.set(resultIndex, operand.get(operandIndex));
  }
  return result;
}

Tensor evalPartitionIdOp(Process *process, MLIRContext *context) {
  if (!process)
    llvm::report_fatal_error(
        "partition_id is only supported when run via interpreter.run_parallel");
  auto partitionId = process->getId().partitionId;
  auto elementType = IntegerType::get(context, 32, IntegerType::Unsigned);
  return makeScalar(Element(elementType, APInt(32, partitionId)));
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
                                 Process *process, Scope &scope,
                                 ArrayRef<ShapedType> resultTypes) {
  SmallVector<Tensor> results;
  for (auto [resultType, initValue] : llvm::zip(resultTypes, initValues))
    results.push_back(makeSplat(resultType, initValue.get({})));

  for (auto inputIt = inputs[0].index_begin(); inputIt != inputs[0].index_end();
       ++inputIt) {
    Index resultIndex;
    for (auto [inputAxis, inputIndexElement] : llvm::enumerate(*inputIt)) {
      if (llvm::is_contained(dimensions, inputAxis)) continue;
      resultIndex.push_back(inputIndexElement);
    }

    SmallVector<InterpreterValue> bodyArgs;
    for (auto [result, initValue] : llvm::zip(results, initValues))
      bodyArgs.push_back(
          makeSplat(initValue.getType(), result.get(resultIndex)));
    for (auto [input, initValue] : llvm::zip(inputs, initValues))
      bodyArgs.emplace_back(
          makeSplat(initValue.getType(), input.get(*inputIt)));

    auto bodyResult = eval(body, bodyArgs, process, &scope);
    for (auto [result, value] : llvm::zip(results, bodyResult))
      result.set(resultIndex, value.getTensor().get({}));
  }
  return results;
}

Tensor evalReducePrecisionOp(const Tensor &operand, int32_t exponentBits,
                             int32_t mantissaBits, ShapedType resultType) {
  Tensor result(resultType);
  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it,
               reducePrecision(operand.get(*it), exponentBits, mantissaBits));
  return result;
}

SmallVector<Tensor> evalReduceWindowOp(
    ArrayRef<Tensor> inputs, ArrayRef<Tensor> initValues,
    const Sizes &windowDimensions, const Sizes &windowStrides,
    const Sizes &baseDilations, const Sizes &windowDilations,
    const Sizes &paddingLow, const Sizes &paddingHigh, Region &body,
    Process *process, Scope &scope, ArrayRef<ShapedType> resultTypes) {
  SmallVector<Tensor> results;
  for (auto [resultType, initValue] : llvm::zip(resultTypes, initValues))
    results.push_back(makeSplat(resultType, initValue.get({})));

  SmallVector<Tensor> paddedInputs;
  for (auto [input, initValue] : llvm::zip(inputs, initValues))
    paddedInputs.push_back(evalPadOp(input, initValue, paddingLow, paddingHigh,
                                     baseDilations - 1));
  for (auto resultIt = results[0].index_begin();
       resultIt != results[0].index_end(); ++resultIt) {
    SmallVector<Tensor> windows;
    auto windowStart = (*resultIt) * windowStrides;
    auto windowEnd = windowStart + (windowDimensions - 1) * windowDilations + 1;
    for (const auto &paddedInput : paddedInputs)
      windows.push_back(
          evalSliceOp(paddedInput, windowStart, windowEnd, windowDilations));

    auto reducedValues = evalReduceOp(windows, initValues, inputs[0].getAxes(),
                                      body, process, scope);
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

Tensor evalReplicaIdOp(Process *process, MLIRContext *context) {
  if (!process)
    llvm::report_fatal_error(
        "replica_id is only supported when run via interpreter.run_parallel");
  auto replicaId = process->getId().replicaId;
  auto elementType = IntegerType::get(context, 32, IntegerType::Unsigned);
  return makeScalar(Element(elementType, APInt(32, replicaId)));
}

Tensor evalReshapeOp(const Tensor &operand, ShapedType resultType) {
  Tensor result(resultType);
  for (auto resultIt = result.index_begin(), operandIt = operand.index_begin();
       resultIt != result.index_end(); ++resultIt, ++operandIt) {
    auto resultIndex = *resultIt;
    auto operandIndex = *operandIt;
    result.set(resultIndex, operand.get(operandIndex));
  }
  return result;
}

Tensor evalReverseOp(const Tensor &operand, const Axes &dimensions,
                     ShapedType resultType) {
  Tensor result(resultType);
  for (auto resultIt = result.index_begin(); resultIt != result.index_end();
       ++resultIt) {
    auto resultIndex = *resultIt;
    Index operandIndex(resultIndex);
    for (auto d : dimensions)
      operandIndex[d] = result.getShape()[d] - operandIndex[d] - 1;
    result.set(resultIndex, operand.get(operandIndex));
  }
  return result;
}

Tensor evalRoundNearestEvenOp(const Tensor &operand, ShapedType resultType) {
  Tensor result(resultType);
  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it, roundNearestEven(operand.get(*it)));
  return result;
}

Tensor evalRoundOp(const Tensor &operand, ShapedType resultType) {
  Tensor result(resultType);
  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it, roundNearestAfz(operand.get(*it)));
  return result;
}

Tensor evalRsqrtOp(const Tensor &operand, ShapedType resultType) {
  Tensor result(resultType);
  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it, rsqrt(operand.get(*it)));
  return result;
}

SmallVector<Tensor> evalScatterOp(
    ArrayRef<Tensor> inputs, const Tensor &scatterIndices,
    ArrayRef<Tensor> updates, const Axes &updateWindowDims,
    const Axes &insertedWindowDims, const Axes &scatterDimsToOperandDims,
    Axis indexVectorDim, Region &updateComputation, Process *process,
    Scope &scope, ArrayRef<ShapedType> resultTypes) {
  SmallVector<Tensor> results;
  for (const auto &input : inputs) results.push_back(input);

  Axes updateScatterDims;
  for (auto d : updates[0].getAxes())
    if (!llvm::is_contained(updateWindowDims, d))
      updateScatterDims.push_back(d);

  for (auto updateIndexIt = updates[0].index_begin();
       updateIndexIt != updates[0].index_end(); ++updateIndexIt) {
    auto updateIndex = *updateIndexIt;
    Index updateScatterIndex;
    for (auto d : updateScatterDims)
      updateScatterIndex.push_back(updateIndex[d]);

    auto startIndicesIndex = updateScatterIndex;
    if (indexVectorDim < scatterIndices.getRank())
      startIndicesIndex.insert(startIndicesIndex.begin() + indexVectorDim,
                               kColon);
    auto startIndex = evalIndex(evalSliceOp(scatterIndices, startIndicesIndex));

    Index fullStartIndex(inputs[0].getRank(), 0);
    for (auto dInput : inputs[0].getAxes()) {
      auto dStartIt = llvm::find(scatterDimsToOperandDims, dInput);
      if (dStartIt == scatterDimsToOperandDims.end()) continue;
      auto dStart = dStartIt - scatterDimsToOperandDims.begin();
      fullStartIndex[dInput] = startIndex[dStart];
    }

    Index updateWindowIndex;
    for (auto d : updateWindowDims) updateWindowIndex.push_back(updateIndex[d]);

    Index fullWindowIndex(updateWindowIndex.size() + insertedWindowDims.size(),
                          0);
    for (size_t i = 0, wi = 0; i < fullWindowIndex.size(); ++i) {
      if (llvm::is_contained(insertedWindowDims, i)) continue;
      fullWindowIndex[i] = updateWindowIndex[wi++];
    }

    auto resultIndex = fullStartIndex + fullWindowIndex;
    if (!resultIndex.inBounds(results[0].getShape())) continue;

    SmallVector<InterpreterValue> updateComputationArgs;
    for (const auto &result : results)
      updateComputationArgs.push_back(makeScalar(result.get(resultIndex)));
    for (const auto &update : updates)
      updateComputationArgs.push_back(makeScalar(update.get(updateIndex)));

    auto updatedValues =
        eval(updateComputation, updateComputationArgs, process, &scope);
    for (auto [result, updatedValue] : llvm::zip(results, updatedValues))
      result.set(resultIndex, updatedValue.getTensor().get({}));
  }

  return results;
}

Tensor evalSelectAndScatterOp(const Tensor &operand, const Tensor &source,
                              const Tensor &initValue,
                              const Sizes &windowDimensions,
                              const Sizes &windowStrides,
                              const Sizes &paddingLow, Region &select,
                              Region &scatter, Process *process, Scope &scope,
                              ShapedType resultType) {
  auto result = makeSplat(resultType, initValue.get({}));

  for (auto sourceIt = source.index_begin(); sourceIt != source.index_end();
       ++sourceIt) {
    std::optional<Element> selectedVal;
    std::optional<Index> selectedIndex;
    auto iterateThroughWindow = [&](std::function<void(const Index &)> body) {
      for (auto windowIt = windowDimensions.index_begin();
           windowIt != windowDimensions.index_end(); ++windowIt) {
        auto operandIndex = *sourceIt * windowStrides + *windowIt - paddingLow;
        if (!operandIndex.inBounds(operand.getShape())) continue;
        body(operandIndex);
      }
    };
    iterateThroughWindow([&](const Index &operandIndex) {
      auto currVal = operand.get(operandIndex);
      if (!selectedVal) {
        selectedVal = currVal;
        selectedIndex = operandIndex;
      }

      InterpreterValue selectedInterpreterVal(makeScalar(selectedVal.value()));
      InterpreterValue currInterpreterVal(makeScalar(currVal));
      auto selectResult =
          eval(select, {selectedInterpreterVal, currInterpreterVal}, process,
               &scope);

      bool selected = !selectResult[0].getTensor().get({}).getBooleanValue();
      if (selected) {
        selectedVal = currVal;
        selectedIndex = operandIndex;
      }
    });
    iterateThroughWindow([&](const Index &operandIndex) {
      if (operandIndex == selectedIndex) {
        Tensor sourceValues(
            RankedTensorType::get({2}, initValue.getElementType()));
        sourceValues.set({0}, source.get(*sourceIt));
        sourceValues.set({1}, result.get(operandIndex));
        auto reducedResult = evalReduceOp({sourceValues}, {initValue}, {0},
                                          scatter, process, scope);
        result.set(operandIndex, reducedResult[0].get({}));
      }
    });
  }
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
    auto resultIndex = *resultIt;
    auto operandIndex = startIndices + resultIndex * strides;
    result.set(resultIndex, operand.get(operandIndex));
  }
  return result;
}

SmallVector<Tensor> evalSortOp(ArrayRef<Tensor> inputs, Axis dimension,
                               bool isStable, Region &comparator,
                               Process *process, Scope &scope) {
  SmallVector<Tensor> results;
  for (const auto &input : inputs) results.emplace_back(input.getType());
  auto adjustedDimension =
      dimension >= 0 ? dimension : dimension + inputs[0].getRank();

  for (auto resultIt = results[0].index_begin();
       resultIt != results[0].index_end(); ++resultIt) {
    // resultIt iterates through all indices in the index space, but sorting
    // only needs to be done once per slice.
    if ((*resultIt)[adjustedDimension] != 0) continue;

    // SortOp sorts 1-dimensional slices of inputs together and produces
    // 1-dimensional slices of results.
    // In this implementation, we aren't going to materialize these slices as
    // a tensor of tuples, but are going to represent these tuples with integer
    // handles, with each handle being an index within the slice.
    // Then, instead of sorting a tensor of tuples, we'll be sorting a tensor of
    // handles, with the comparator knowing how to use these handles to fetch
    // the actual input elements being compared.
    SmallVector<int64_t> inputsTogether(
        inputs[0].getShape()[adjustedDimension]);
    std::iota(inputsTogether.begin(), inputsTogether.end(), 0);
    auto comparatorTogether = [&](int64_t lhsHandle, int64_t rhsHandle) {
      SmallVector<InterpreterValue> args;
      auto lhsIndex = *resultIt;
      auto rhsIndex = *resultIt;
      lhsIndex[adjustedDimension] = lhsHandle;
      rhsIndex[adjustedDimension] = rhsHandle;
      for (const auto &input : inputs) {
        args.emplace_back(makeScalar(input.get(lhsIndex)));
        args.emplace_back(makeScalar(input.get(rhsIndex)));
      }
      auto comparatorResult = eval(comparator, args, process, &scope);
      return comparatorResult[0].getTensor().get({}).getBooleanValue();
    };
    if (isStable)
      std::stable_sort(inputsTogether.begin(), inputsTogether.end(),
                       comparatorTogether);
    else
      std::sort(inputsTogether.begin(), inputsTogether.end(),
                comparatorTogether);

    // After the tensor of handles has been sorted, we apply the results of
    // this sort by reshuffling input elements into result elements.
    auto &resultsTogether = inputsTogether;
    for (auto [inputHandle, resultHandle] : llvm::enumerate(resultsTogether)) {
      for (auto [input, result] : llvm::zip(inputs, results)) {
        auto inputIndex = *resultIt;
        auto resultIndex = *resultIt;
        inputIndex[adjustedDimension] = inputHandle;
        resultIndex[adjustedDimension] = resultHandle;
        result.set(resultIndex, input.get(inputIndex));
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
    auto operandIndex = *operandIt;
    Index resultIndex(result.getRank());
    for (auto d = 0; d < result.getRank(); d++)
      resultIndex[d] = operandIndex[permutation[d]];
    result.set(resultIndex, operand.get(operandIndex));
  }
  return result;
}

SmallVector<InterpreterValue> evalWhileOp(SmallVector<InterpreterValue> operand,
                                          Region &cond, Region &body,
                                          Process *process, Scope &scope) {
  SmallVector<InterpreterValue> results(operand);

  auto condResults = eval(cond, operand, process, &scope);

  while (condResults[0].getTensor().get({}).getBooleanValue()) {
    results = eval(body, results, process, &scope);
    condResults = eval(cond, results, process, &scope);
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
