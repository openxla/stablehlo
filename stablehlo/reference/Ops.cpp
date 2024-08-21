/* Copyright 2023-2024 The StableHLO Authors.

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
#include <cstddef>
#include <cstdint>
#include <functional>
#include <numeric>
#include <optional>
#include <utility>

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "stablehlo/dialect/Base.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/dialect/TypeInference.h"
#include "stablehlo/reference/Axes.h"
#include "stablehlo/reference/Configuration.h"
#include "stablehlo/reference/Element.h"
#include "stablehlo/reference/Errors.h"
#include "stablehlo/reference/Index.h"
#include "stablehlo/reference/Process.h"
#include "stablehlo/reference/ProcessGrid.h"
#include "stablehlo/reference/Scope.h"
#include "stablehlo/reference/Tensor.h"
#include "stablehlo/reference/Token.h"
#include "stablehlo/reference/Types.h"
#include "stablehlo/reference/Value.h"

#define DEBUG_TYPE "stablehlo-interpreter"

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

template <typename T>
SmallVector<T> extractAttributeOrDefault(std::optional<ArrayRef<T>> attr,
                                         int64_t size, T value) {
  if (attr.has_value()) return llvm::to_vector(attr.value());
  return SmallVector<T>(size, value);
}

Tensor dotGeneralOp(const Tensor &lhs, const Tensor &rhs,
                    const Axes &lhsContractingDimensions,
                    const Axes &rhsContractingDimensions) {
  SmallVector<ShapedTypeComponents> inferredDotGeneralType;
  if (failed(hlo::inferDotGeneralOp(
          /*location=*/{}, lhs.getType(), rhs.getType(),
          /*lhsBatchingDimensions=*/{}, /*rhsBatchingDimensions*/ {},
          lhsContractingDimensions, rhsContractingDimensions,
          /*precisionConfig=*/{}, inferredDotGeneralType)))
    report_fatal_error(
        invalidArgument("Could not infer DotGeneralOp's return type"));

  return dotGeneralOp(lhs, rhs, /*lhsBatchingDimensions=*/{},
                      /*rhsBatchingDimensions*/ {}, lhsContractingDimensions,
                      rhsContractingDimensions,
                      RankedTensorType::get(inferredDotGeneralType[0].getDims(),
                                            lhs.getElementType()));
}

Tensor padOp(const Tensor &operand, const Tensor &paddingValue,
             const Sizes &edgePaddingLow, const Sizes &edgePaddingHigh,
             const Sizes &interiorPadding) {
  SmallVector<Type> inferredTypes;
  Builder builder(operand.getType().getContext());
  auto inferStatus = hlo::inferPadOp(
      {}, operand.getType(), paddingValue.getType(), edgePaddingLow,
      edgePaddingHigh, interiorPadding, inferredTypes);
  if (failed(inferStatus))
    report_fatal_error(invalidArgument("Could not infer PadOp's return type"));
  return padOp(operand, paddingValue, edgePaddingLow, interiorPadding,
               cast<ShapedType>(inferredTypes[0]));
}

SmallVector<Tensor> reduceOp(ArrayRef<Tensor> inputs,
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
      /*location=*/{}, inputTypes, dimensions, body, inferredReduceTypes);
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
  return reduceOp(inputs, initValues, dimensions, body, process, scope,
                  resultTypes);
}

Tensor sliceOp(const Tensor &operand, const Sizes &startIndices,
               const Sizes &limitIndices, const Sizes &strides) {
  SmallVector<Type> inferredTypes;
  Builder builder(operand.getType().getContext());
  auto inferStatus = hlo::inferSliceOp({}, operand.getType(), startIndices,
                                       limitIndices, strides, inferredTypes);
  if (failed(inferStatus))
    report_fatal_error(
        invalidArgument("Could not infer SliceOp's return type"));
  return sliceOp(operand, startIndices, strides,
                 cast<ShapedType>(inferredTypes[0]));
}

SmallVector<InterpreterValue> callOp(ArrayRef<Tensor> inputs,
                                     InterpreterFallback *fallback,
                                     Process *process, Operation *op,
                                     StringRef funcName) {
  SymbolTableCollection symbolTableCollection;
  auto symbolTable =
      symbolTableCollection.getSymbolTable(op->getParentOfType<ModuleOp>());
  auto func = symbolTable.lookupNearestSymbolFrom<func::FuncOp>(
      op, StringAttr::get(op->getContext(), funcName));
  SmallVector<InterpreterValue> values = llvm::map_to_vector(
      inputs, [](const Tensor &t) { return InterpreterValue(t); });
  return eval(func.getBody(), values, fallback, process, nullptr);
}

// Experimental notation for slices, roughly following the spec notation.
// TODO(#1401): Might evolve in the future together with the spec.
constexpr int64_t kColon = -1;
Tensor sliceOp(const Tensor &operand, const Index &index) {
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
  return sliceOp(operand, start, limit, strides);
}

Sizes extractElements(ArrayRef<int64_t> arr, ArrayRef<int64_t> indices) {
  Sizes elements;
  for (int64_t index : indices) elements.push_back(arr[index]);
  return elements;
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

template <typename T>
DenseIntElementsAttr getDenseIntElementsAttr(Type elementType, T values,
                                             SmallVector<int64_t> valuesShape) {
  return DenseIntElementsAttr::get(
      RankedTensorType::get(valuesShape, elementType), values);
}

SmallVector<SmallVector<uint32_t>> getReplicaGroups(
    DenseIntElementsAttr replicaGroupsAttr) {
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
  return replicaGroups;
}

Tensor convolutionOp(
    const Tensor &lhs, const Tensor &rhs, ArrayRef<int64_t> windowStrides,
    ArrayRef<std::pair<int64_t, int64_t>> padding,
    ArrayRef<int64_t> lhsDilation, ArrayRef<int64_t> rhsDilation,
    ArrayRef<bool> windowReversal, Axis inputBatchDimension,
    Axis inputFeatureDimension, const Axes &inputSpatialDimensions,
    Axis kernelInputFeatureDimension, Axis kernelOutputFeatureDimension,
    const Axes &kernelSpatialDimensions, Axis outputBatchDimension,
    Axis outputFeatureDimension, const Axes &outputSpatialDimensions,
    int64_t featureGroupCount, int64_t batchGroupCount,
    std::optional<ArrayAttr> precisionConfig, ShapedType resultType) {
  SmallVector<int64_t> paddingVector;
  for (auto pair : padding) {
    paddingVector.push_back(pair.first);
    paddingVector.push_back(pair.second);
  }

  SmallVector<ShapedTypeComponents> inferredConvolutionType;
  if (failed(hlo::inferConvolutionOp(
          /*location=*/{}, lhs.getType(), rhs.getType(), windowStrides,
          /*padding=*/
          getDenseIntElementsAttr(
              IntegerType::get(lhs.getType().getContext(), 64), paddingVector,
              SmallVector<int64_t>({lhs.getRank() - 2, 2})),
          lhsDilation, rhsDilation, windowReversal, inputBatchDimension,
          inputFeatureDimension, ArrayRef<int64_t>(inputSpatialDimensions),
          kernelInputFeatureDimension, kernelOutputFeatureDimension,
          ArrayRef<int64_t>(kernelSpatialDimensions), outputBatchDimension,
          outputFeatureDimension, ArrayRef<int64_t>(outputSpatialDimensions),
          featureGroupCount, batchGroupCount,
          /*precisionConfig=*/{}, inferredConvolutionType)))
    report_fatal_error(
        invalidArgument("Could not infer ConvolutionOp's return type"));

  return convolutionOp(
      lhs, rhs, windowStrides, padding, lhsDilation, rhsDilation,
      windowReversal, inputBatchDimension, inputFeatureDimension,
      inputSpatialDimensions, kernelInputFeatureDimension,
      kernelOutputFeatureDimension, kernelSpatialDimensions,
      outputBatchDimension, outputFeatureDimension, outputSpatialDimensions,
      featureGroupCount, batchGroupCount,
      RankedTensorType::get(inferredConvolutionType[0].getDims(),
                            resultType.getElementType()));
}

// Returns `result` with the effect of applying `permutation`
// (= [dimA] + dimsB + [dimC]) to `input` (= [n] + hw + [c]) such that
// result[permutation[i]] = input[i].
template <typename T>
SmallVector<T> concatAndPermute(T n, SmallVector<T> hw, T c,
                                const Axes &permutation) {
  SmallVector<T> result(permutation.size());
  result[permutation[0]] = n;
  result[permutation[permutation.size() - 1]] = c;
  for (uint64_t i = 1; i < permutation.size() - 1; ++i)
    result[permutation[i]] = hw[i - 1];
  return result;
}

Tensor constant(Element initValue) {
  Tensor result(RankedTensorType::get({}, initValue.getType()));
  result.set({}, initValue);
  return result;
}

template <typename T>
Tensor constant(T value, Type elementType) {
  return constant(convert(elementType, value));
}

Tensor makeSplat(ShapedType type, const Element &initValue) {
  Tensor result(type);
  for (auto indexIt = result.index_begin(); indexIt != result.index_end();
       ++indexIt)
    result.set(*indexIt, initValue);
  return result;
}

SmallVector<Tensor> split(const Tensor &x, int64_t numResults, Axis axis,
                          MLIRContext *context) {
  Sizes resultShape(x.getShape());
  if (resultShape[axis] % numResults != 0)
    report_fatal_error(
        invalidArgument("input dimension at axis (%d) should be divisible "
                        "by numResults (%d), but got: %d",
                        axis, numResults, resultShape[axis]));

  resultShape[axis] /= numResults;

  SmallVector<Tensor> results;
  for (auto i = 0; i < numResults; ++i) {
    SmallVector<Tensor> inputStartIndices(
        x.getRank(), constant(0.0, IntegerType::get(context, 64)));
    inputStartIndices[axis] =
        constant(i * resultShape[axis], IntegerType::get(context, 64));

    auto result =
        dynamicSliceOp(x, inputStartIndices, resultShape,
                       RankedTensorType::get(resultShape, x.getElementType()));
    results.push_back(result);
  }
  return results;
}

}  // namespace

SmallVector<InterpreterValue> eval(Region &region,
                                   ArrayRef<InterpreterValue> args,
                                   InterpreterFallback *fallback,
                                   Process *process, Scope *parent) {
  Block &block = region.front();
  if (block.getArguments().size() != args.size())
    report_fatal_error(invalidArgument(
        "Expected same number of block arguments and runtime arguments (%d)",
        args.size()));

  Scope scope(parent);
  scope.add(block.getArguments(), args);

  for (Operation &operation : block) {
    if (!llvm::all_of(operation.getResults(), [](OpResult r) {
          if (auto shaped = dyn_cast<ShapedType>(r.getType()))
            return shaped.hasStaticShape();
          return true;
        }))
      llvm::report_fatal_error(
          "dynamic result types are not supported at the moment");

    if (auto op = dyn_cast<AbsOp>(operation)) {
      auto operand = scope.findTensor(op.getOperand());
      auto result = absOp(operand, op.getType());
      scope.add(op.getResult(), result);
    } else if (auto op = dyn_cast<AddOp>(operation)) {
      auto lhs = scope.findTensor(op.getLhs());
      auto rhs = scope.findTensor(op.getRhs());
      auto result = addOp(lhs, rhs, op.getType());
      scope.add(op.getResult(), result);
    } else if (auto op = dyn_cast<AfterAllOp>(operation)) {
      auto inputs = scope.findTokens(op.getInputs());
      auto result = afterAllOp(inputs, op->getContext());
      scope.add(op.getResult(), result);
    } else if (auto op = dyn_cast<AllGatherOp>(operation)) {
      auto operands = scope.findTensors(op.getOperands());

      auto replicaGroupsAttr = op.getReplicaGroups();
      auto replicaGroupsShape = replicaGroupsAttr.getShapedType().getShape();
      SmallVector<SmallVector<uint32_t>> replicaGroups(replicaGroupsShape[0]);
      auto replicaGroupsIt = replicaGroupsAttr.getValues<int64_t>().begin();
      for (auto &replicaGroup : replicaGroups)
        for (auto i = 0; i < replicaGroupsShape[1]; ++i, ++replicaGroupsIt)
          replicaGroup.push_back(*replicaGroupsIt);

      ChannelId channelId = 0;
      if (auto channelHandle = op.getChannelHandle())
        channelId = channelHandle->getHandle();
      SmallVector<ShapedType> resultTypes(op->getResultTypes());

      auto results =
          allGatherOp(operands, op.getAllGatherDim(), replicaGroups, channelId,
                      op.getUseGlobalDeviceIds(), process, resultTypes);
      scope.add(op.getResults(), results);
    } else if (auto op = dyn_cast<AllReduceOp>(operation)) {
      auto operands = scope.findTensors(op.getOperands());
      auto replicaGroups = getReplicaGroups(op.getReplicaGroups());

      ChannelId channelId = 0;
      if (auto channelHandle = op.getChannelHandle())
        channelId = channelHandle->getHandle();
      SmallVector<ShapedType> resultTypes(op->getResultTypes());

      auto results = allReduceOp(
          operands, replicaGroups, channelId, op.getUseGlobalDeviceIds(),
          op.getComputation(), process, scope, resultTypes);
      scope.add(op.getResults(), results);
    } else if (auto op = dyn_cast<AllToAllOp>(operation)) {
      auto operands = scope.findTensors(op.getOperands());
      auto replicaGroupsAttr = op.getReplicaGroups();
      auto replicaGroupsShape = replicaGroupsAttr.getShapedType().getShape();
      SmallVector<SmallVector<uint32_t>> replicaGroups(replicaGroupsShape[0]);
      auto replicaGroupsIt = replicaGroupsAttr.getValues<int64_t>().begin();
      for (auto &replicaGroup : replicaGroups)
        for (auto i = 0; i < replicaGroupsShape[1]; ++i, ++replicaGroupsIt)
          replicaGroup.push_back(*replicaGroupsIt);

      ChannelId channelId = 0;
      if (auto channelHandle = op.getChannelHandle())
        channelId = channelHandle->getHandle();
      SmallVector<ShapedType> resultTypes(op->getResultTypes());

      auto results = allToAllOp(operands, op.getSplitDimension(),
                                op.getConcatDimension(), op.getSplitCount(),
                                replicaGroups, channelId, process, resultTypes);
      scope.add(op.getResults(), results);
    } else if (auto op = dyn_cast<AndOp>(operation)) {
      auto lhs = scope.findTensor(op.getLhs());
      auto rhs = scope.findTensor(op.getRhs());
      auto result = andOp(lhs, rhs, op.getType());
      scope.add(op.getResult(), result);
    } else if (auto op = dyn_cast<Atan2Op>(operation)) {
      auto lhs = scope.findTensor(op.getLhs());
      auto rhs = scope.findTensor(op.getRhs());
      auto result = atan2Op(lhs, rhs, op.getType());
      scope.add(op.getResult(), result);
    } else if (isa<BatchNormGradOp>(operation)) {
      failOnDecomposableOp(operation);
    } else if (isa<BatchNormInferenceOp>(operation)) {
      failOnDecomposableOp(operation);
    } else if (isa<BatchNormTrainingOp>(operation)) {
      failOnDecomposableOp(operation);
    } else if (auto op = dyn_cast<BitcastConvertOp>(operation)) {
      auto operand = scope.findTensor(op.getOperand());
      auto result = bitcastConvertOp(operand, op.getType());
      scope.add(op.getResult(), result);
    } else if (auto op = dyn_cast<BroadcastInDimOp>(operation)) {
      auto operand = scope.findTensor(op.getOperand());
      auto broadcastDimensions = Axes(op.getBroadcastDimensions());
      auto result =
          broadcastInDimOp(operand, broadcastDimensions, op.getType());
      scope.add(op.getResult(), result);
    } else if (isa<BroadcastOp>(operation)) {
      failOnDecomposableOp(operation);
    } else if (auto op = dyn_cast<func::CallOp>(operation)) {
      auto operands = scope.findTensors(op.getOperands());
      auto results =
          callOp(operands, fallback, process, &operation, op.getCallee());
      scope.add(op.getResults(), results);
    } else if (auto op = dyn_cast<CaseOp>(operation)) {
      auto index = scope.findTensor(op.getIndex());
      auto branches = op.getBranches();
      auto results = caseOp(index, branches, process, scope);
      scope.add(op.getResults(), results);
    } else if (auto op = dyn_cast<CbrtOp>(operation)) {
      auto operand = scope.findTensor(op.getOperand());
      auto result = cbrtOp(operand, op.getType());
      scope.add(op.getResult(), result);
    } else if (auto op = dyn_cast<CeilOp>(operation)) {
      auto operand = scope.findTensor(op.getOperand());
      auto result = ceilOp(operand, op.getType());
      scope.add(op.getResult(), result);
    } else if (isa<CholeskyOp>(operation)) {
      failOnDecomposableOp(operation);
    } else if (auto op = dyn_cast<ClampOp>(operation)) {
      auto min = scope.findTensor(op.getMin());
      auto operand = scope.findTensor(op.getOperand());
      auto max = scope.findTensor(op.getMax());
      auto result = clampOp(min, operand, max, op.getType());
      scope.add(op.getResult(), result);
    } else if (auto op = dyn_cast<ClzOp>(operation)) {
      auto operand = scope.findTensor(op.getOperand());
      auto result = clzOp(operand, op.getType());
      scope.add(op.getResult(), result);
    } else if (auto op = dyn_cast<CollectiveBroadcastOp>(operation)) {
      auto operand = scope.findTensor(op.getOperand());

      auto replicaGroupsAttr = op.getReplicaGroups();
      auto replicaGroupsShape = replicaGroupsAttr.getShapedType().getShape();
      SmallVector<SmallVector<uint32_t>> replicaGroups(replicaGroupsShape[0]);
      auto replicaGroupsIt = replicaGroupsAttr.getValues<int64_t>().begin();
      for (auto &replicaGroup : replicaGroups)
        for (auto i = 0; i < replicaGroupsShape[1]; ++i, ++replicaGroupsIt)
          replicaGroup.push_back(*replicaGroupsIt);

      ChannelId channelId = 0;
      if (auto channelHandle = op.getChannelHandle())
        channelId = channelHandle->getHandle();

      auto result =
          collectiveBroadcastOp(operand, replicaGroups, channelId, process);
      scope.add(op.getResult(), result);
    } else if (auto op = dyn_cast<CollectivePermuteOp>(operation)) {
      auto operand = scope.findTensor(op.getOperand());

      auto sourceTargetPairsAttr = op.getSourceTargetPairs();
      SmallVector<SmallVector<uint32_t>> sourceTargetPairs(
          sourceTargetPairsAttr.getNumElements() / 2);
      auto sourceTargetPairsIt =
          sourceTargetPairsAttr.getValues<int64_t>().begin();
      for (auto &sourceTargetPair : sourceTargetPairs) {
        sourceTargetPair.push_back(*sourceTargetPairsIt++);
        sourceTargetPair.push_back(*sourceTargetPairsIt++);
      }

      ChannelId channelId = 0;
      if (auto channelHandle = op.getChannelHandle())
        channelId = channelHandle->getHandle();

      auto result =
          collectivePermuteOp(operand, sourceTargetPairs, channelId, process);
      scope.add(op.getResult(), result);
    } else if (auto op = dyn_cast<CompareOp>(operation)) {
      auto lhs = scope.findTensor(op.getLhs());
      auto rhs = scope.findTensor(op.getRhs());
      auto comparisonDirection = op.getComparisonDirection();
      auto result = compareOp(lhs, rhs, comparisonDirection, op.getType());
      scope.add(op.getResult(), result);
    } else if (auto op = dyn_cast<ComplexOp>(operation)) {
      auto lhs = scope.findTensor(op.getLhs());
      auto rhs = scope.findTensor(op.getRhs());
      auto result = complexOp(lhs, rhs, op.getType());
      scope.add(op.getResult(), result);
    } else if (auto op = dyn_cast<CompositeOp>(operation)) {
      auto operands = scope.findTensors(op.getOperands());
      auto results = callOp(operands, fallback, process, &operation,
                            op.getDecomposition());
      scope.add(op.getResults(), results);
    } else if (auto op = dyn_cast<ConcatenateOp>(operation)) {
      auto operands = scope.findTensors(op.getOperands());
      auto result = concatenateOp(operands, op.getDimension(), op.getType());
      scope.add(op.getResult(), result);
    } else if (auto op = dyn_cast<ConstantOp>(operation)) {
      auto result = constantOp(op.getValue());
      scope.add(op.getResult(), result);
    } else if (auto op = dyn_cast<ConvertOp>(operation)) {
      auto operand = scope.findTensor(op.getOperand());
      auto result = convertOp(operand, op.getType());
      scope.add(op.getResult(), result);
    } else if (auto op = dyn_cast<ConvolutionOp>(operation)) {
      auto lhs = scope.findTensor(op.getLhs());
      auto rhs = scope.findTensor(op.getRhs());
      auto rank = lhs.getRank();

      SmallVector<int64_t> windowStrides = extractAttributeOrDefault<int64_t>(
          op.getWindowStrides(), rank - 2, 1);

      SmallVector<std::pair<int64_t, int64_t>> padding(rank - 2, {0, 0});
      if (auto paddingAttr = op.getPaddingAttr()) {
        auto paddingOrErr = hlo::convertPaddingAttribute(paddingAttr, {});
        if (failed(paddingOrErr))
          report_fatal_error(invalidArgument("Invalid padding format found."));
        padding = *paddingOrErr;
      }

      SmallVector<int64_t> lhsDilation(rank - 2, 1);
      if (auto lhsDilationAttr = op.getLhsDilation())
        lhsDilation = SmallVector<int64_t>(lhsDilationAttr.value());

      SmallVector<int64_t> rhsDilation(rank - 2, 1);
      if (auto rhsDilationAttr = op.getRhsDilation())
        rhsDilation = SmallVector<int64_t>(rhsDilationAttr.value());

      SmallVector<bool> windowReversal(rank - 2, false);
      if (auto windowReversalAttr = op.getWindowReversal())
        windowReversal = SmallVector<bool>(windowReversalAttr.value());

      auto dimensionNumbers = op.getDimensionNumbers();
      auto result = convolutionOp(
          lhs, rhs, windowStrides, padding, lhsDilation, rhsDilation,
          windowReversal, dimensionNumbers.getInputBatchDimension(),
          dimensionNumbers.getInputFeatureDimension(),
          Axes(dimensionNumbers.getInputSpatialDimensions()),
          dimensionNumbers.getKernelInputFeatureDimension(),
          dimensionNumbers.getKernelOutputFeatureDimension(),
          Axes(dimensionNumbers.getKernelSpatialDimensions()),
          dimensionNumbers.getOutputBatchDimension(),
          dimensionNumbers.getOutputFeatureDimension(),
          Axes(dimensionNumbers.getOutputSpatialDimensions()),
          op.getFeatureGroupCount(), op.getBatchGroupCount(), op.getType());
      scope.add(op.getResult(), result);
    } else if (auto op = dyn_cast<CosineOp>(operation)) {
      auto operand = scope.findTensor(op.getOperand());
      auto result = cosineOp(operand, op.getType());
      scope.add(op.getResult(), result);
    } else if (isa<CreateTokenOp>(operation)) {
      failOnDecomposableOp(operation);
    } else if (isa<CrossReplicaSumOp>(operation)) {
      failOnDecomposableOp(operation);
    } else if (auto op = dyn_cast<DivOp>(operation)) {
      auto lhs = scope.findTensor(op.getLhs());
      auto rhs = scope.findTensor(op.getRhs());
      auto result = divideOp(lhs, rhs, op.getType());
      scope.add(op.getResult(), result);
    } else if (isa<DotOp>(operation)) {
      failOnDecomposableOp(operation);
    } else if (auto op = dyn_cast<DotGeneralOp>(operation)) {
      LLVM_DEBUG({
        if (op.getAlgorithm().has_value())
          llvm::dbgs() << "ignoring dot algorithm constraints in interpreter";
      });
      auto lhs = scope.findTensor(op.getLhs());
      auto rhs = scope.findTensor(op.getRhs());
      auto lhsBatchingDimensions =
          Axes(op.getDotDimensionNumbers().getLhsBatchingDimensions());
      auto rhsBatchingDimensions =
          Axes(op.getDotDimensionNumbers().getRhsBatchingDimensions());
      auto lhsContractingDimensions =
          Axes(op.getDotDimensionNumbers().getLhsContractingDimensions());
      auto rhsContractingDimensions =
          Axes(op.getDotDimensionNumbers().getRhsContractingDimensions());
      auto result = dotGeneralOp(
          lhs, rhs, lhsBatchingDimensions, rhsBatchingDimensions,
          lhsContractingDimensions, rhsContractingDimensions, op.getType());
      scope.add(op.getResult(), result);
    } else if (auto op = dyn_cast<DynamicBroadcastInDimOp>(operation)) {
      auto operand = scope.findTensor(op.getOperand());
      auto broadcastDimensions = Axes(op.getBroadcastDimensions());
      auto result =
          broadcastInDimOp(operand, broadcastDimensions, op.getType());
      scope.add(op.getResult(), result);
    } else if (auto op = dyn_cast<DynamicConvOp>(operation)) {
      auto lhs = scope.findTensor(op.getLhs());
      auto rhs = scope.findTensor(op.getRhs());
      auto dPadding = scope.findTensor(op.getPadding());
      auto rank = lhs.getRank();

      SmallVector<int64_t> windowStrides(rank - 2, 1);
      if (auto windowStridesAttr = op.getWindowStrides())
        windowStrides = SmallVector<int64_t>(windowStridesAttr.value());

      SmallVector<int64_t> lhsDilation(rank - 2, 1);
      if (auto lhsDilationAttr = op.getLhsDilation())
        lhsDilation = SmallVector<int64_t>(lhsDilationAttr.value());

      SmallVector<int64_t> rhsDilation(rank - 2, 1);
      if (auto rhsDilationAttr = op.getRhsDilation())
        rhsDilation = SmallVector<int64_t>(rhsDilationAttr.value());

      SmallVector<bool> windowReversal(rank - 2, false);
      if (auto windowReversalAttr = op.getWindowReversal())
        windowReversal = SmallVector<bool>(windowReversalAttr.value());

      auto dimensionNumbers = op.getDimensionNumbers();
      SmallVector<std::pair<int64_t, int64_t>> padding;
      for (auto it = dPadding.index_begin(); it != dPadding.index_end(); ++it) {
        auto paddingLow = dPadding.get(*it).getIntegerValue().getSExtValue();
        auto paddingHigh =
            dPadding.get(*(++it)).getIntegerValue().getSExtValue();
        padding.push_back({paddingLow, paddingHigh});
      }
      auto result = convolutionOp(
          lhs, rhs, windowStrides, padding, lhsDilation, rhsDilation,
          windowReversal, dimensionNumbers.getInputBatchDimension(),
          dimensionNumbers.getInputFeatureDimension(),
          Axes(dimensionNumbers.getInputSpatialDimensions()),
          dimensionNumbers.getKernelInputFeatureDimension(),
          dimensionNumbers.getKernelOutputFeatureDimension(),
          Axes(dimensionNumbers.getKernelSpatialDimensions()),
          dimensionNumbers.getOutputBatchDimension(),
          dimensionNumbers.getOutputFeatureDimension(),
          Axes(dimensionNumbers.getOutputSpatialDimensions()),
          op.getFeatureGroupCount(), op.getBatchGroupCount(), op.getType());
      scope.add(op.getResult(), result);
    } else if (auto op = dyn_cast<DynamicGatherOp>(operation)) {
      auto operand = scope.findTensor(op.getOperand());
      auto startIndices = scope.findTensor(op.getStartIndices());
      auto sliceSizes = scope.findTensor(op.getSliceSizes());
      auto result = gatherOp(
          operand, startIndices, Axes(op.getDimensionNumbers().getOffsetDims()),
          Axes(op.getDimensionNumbers().getCollapsedSliceDims()),
          Axes(op.getDimensionNumbers().getOperandBatchingDims()),
          Axes(op.getDimensionNumbers().getStartIndicesBatchingDims()),
          Axes(op.getDimensionNumbers().getStartIndexMap()),
          Axis(op.getDimensionNumbers().getIndexVectorDim()),
          makeSizes(sliceSizes), op.getIndicesAreSorted(), op.getType());
      scope.add(op.getResult(), result);
    } else if (auto op = dyn_cast<DynamicIotaOp>(operation)) {
      auto iotaDimension = op.getIotaDimension();
      auto result = iotaOp(iotaDimension, op.getType());
      scope.add(op.getResult(), result);
    } else if (auto op = dyn_cast<DynamicPadOp>(operation)) {
      auto operand = scope.findTensor(op.getOperand());
      auto paddingValue = scope.findTensor(op.getPaddingValue());
      auto edgePaddingLow = scope.findTensor(op.getEdgePaddingLow());
      auto edgePaddingHigh = scope.findTensor(op.getEdgePaddingHigh());
      auto interiorPadding = scope.findTensor(op.getInteriorPadding());
      auto result =
          padOp(operand, paddingValue, makeSizes(edgePaddingLow),
                makeSizes(edgePaddingHigh), makeSizes(interiorPadding));
      scope.add(op.getResult(), result);
    } else if (auto op = dyn_cast<DynamicReshapeOp>(operation)) {
      auto operand = scope.findTensor(op.getOperand());
      auto result = reshapeOp(operand, op.getType());
      scope.add(op.getResult(), result);
    } else if (auto op = dyn_cast<DynamicSliceOp>(operation)) {
      auto operand = scope.findTensor(op.getOperand());
      auto startIndices = scope.findTensors(op.getStartIndices());
      auto sliceSizes = Sizes(op.getSliceSizes());
      auto result =
          dynamicSliceOp(operand, startIndices, sliceSizes, op.getType());
      scope.add(op.getResult(), result);
    } else if (auto op = dyn_cast<DynamicUpdateSliceOp>(operation)) {
      auto operand = scope.findTensor(op.getOperand());
      auto update = scope.findTensor(op.getUpdate());
      auto startIndices = scope.findTensors(op.getStartIndices());
      auto result =
          dynamicUpdateSliceOp(operand, update, startIndices, op.getType());
      scope.add(op.getResult(), result);
    } else if (isa<EinsumOp>(operation)) {
      failOnDecomposableOp(operation);
    } else if (auto op = dyn_cast<ExpOp>(operation)) {
      auto operand = scope.findTensor(op.getOperand());
      auto result = exponentialOp(operand, op.getType());
      scope.add(op.getResult(), result);
    } else if (auto op = dyn_cast<Expm1Op>(operation)) {
      auto operand = scope.findTensor(op.getOperand());
      auto result = expm1Op(operand, op.getType());
      scope.add(op.getResult(), result);
    } else if (auto op = dyn_cast<FloorOp>(operation)) {
      auto operand = scope.findTensor(op.getOperand());
      auto result = floorOp(operand, op.getType());
      scope.add(op.getResult(), result);
    } else if (auto op = dyn_cast<GatherOp>(operation)) {
      auto operand = scope.findTensor(op.getOperand());
      auto startIndices = scope.findTensor(op.getStartIndices());
      auto result = gatherOp(
          operand, startIndices, Axes(op.getDimensionNumbers().getOffsetDims()),
          Axes(op.getDimensionNumbers().getCollapsedSliceDims()),
          Axes(op.getDimensionNumbers().getOperandBatchingDims()),
          Axes(op.getDimensionNumbers().getStartIndicesBatchingDims()),
          Axes(op.getDimensionNumbers().getStartIndexMap()),
          Axis(op.getDimensionNumbers().getIndexVectorDim()),
          Sizes(op.getSliceSizes()), op.getIndicesAreSorted(), op.getType());
      scope.add(op.getResult(), result);
    } else if (auto op = dyn_cast<GetDimensionSizeOp>(operation)) {
      auto operand = scope.findTensor(op.getOperand());
      auto dimension = op.getDimension();
      auto result = getDimensionSizeOp(operand, dimension, op.getType());
      scope.add(op.getResult(), result);
    } else if (auto op = dyn_cast<GetTupleElementOp>(operation)) {
      auto operand = scope.findTuple(op.getOperand());
      auto result = getTupleElementOp(operand, op.getIndex());
      scope.add(op.getResult(), result);
    } else if (auto op = dyn_cast<IfOp>(operation)) {
      auto pred = scope.findTensor(op.getPred());
      auto &trueBranch = op.getTrueBranch();
      auto &falseBranch = op.getFalseBranch();
      auto results = ifOp(pred, trueBranch, falseBranch, process, scope);
      scope.add(op.getResults(), results);
    } else if (auto op = dyn_cast<ImagOp>(operation)) {
      auto operand = scope.findTensor(op.getOperand());
      auto result = imagOp(operand, op.getType());
      scope.add(op.getResult(), result);
    } else if (auto op = dyn_cast<InfeedOp>(operation)) {
      auto token = scope.findToken(op.getToken());
      auto results = infeedOp(token, process, region, scope);
      scope.add(op.getResults(), results);
    } else if (auto op = dyn_cast<IotaOp>(operation)) {
      auto iotaDimension = op.getIotaDimension();
      auto result = iotaOp(iotaDimension, op.getType());
      scope.add(op.getResult(), result);
    } else if (auto op = dyn_cast<IsFiniteOp>(operation)) {
      auto operand = scope.findTensor(op.getOperand());
      auto result = isFiniteOp(operand, op.getType());
      scope.add(op.getResult(), result);
    } else if (auto op = dyn_cast<Log1pOp>(operation)) {
      auto operand = scope.findTensor(op.getOperand());
      auto result = log1pOp(operand, op.getType());
      scope.add(op.getResult(), result);
    } else if (auto op = dyn_cast<LogOp>(operation)) {
      auto operand = scope.findTensor(op.getOperand());
      auto result = logOp(operand, op.getType());
      scope.add(op.getResult(), result);
    } else if (auto op = dyn_cast<LogisticOp>(operation)) {
      auto operand = scope.findTensor(op.getOperand());
      auto result = logisticOp(operand, op.getType());
      scope.add(op.getResult(), result);
    } else if (auto op = dyn_cast<MapOp>(operation)) {
      auto inputs = scope.findTensors(op.getInputs());
      auto &computation = op.getComputation();
      auto result = mapOp(inputs, computation, process, scope, op.getType());
      scope.add(op.getResult(), result);
    } else if (auto op = dyn_cast<MaxOp>(operation)) {
      auto lhs = scope.findTensor(op.getLhs());
      auto rhs = scope.findTensor(op.getRhs());
      auto result = maxOp(lhs, rhs, op.getType());
      scope.add(op.getResult(), result);
    } else if (auto op = dyn_cast<MinOp>(operation)) {
      auto lhs = scope.findTensor(op.getLhs());
      auto rhs = scope.findTensor(op.getRhs());
      auto result = minOp(lhs, rhs, op.getType());
      scope.add(op.getResult(), result);
    } else if (auto op = dyn_cast<MulOp>(operation)) {
      auto lhs = scope.findTensor(op.getLhs());
      auto rhs = scope.findTensor(op.getRhs());
      auto result = multiplyOp(lhs, rhs, op.getType());
      scope.add(op.getResult(), result);
    } else if (auto op = dyn_cast<NegOp>(operation)) {
      auto operand = scope.findTensor(op.getOperand());
      auto result = negOp(operand, op.getType());
      scope.add(op.getResult(), result);
    } else if (auto op = dyn_cast<NotOp>(operation)) {
      auto operand = scope.findTensor(op.getOperand());
      auto result = notOp(operand, op.getType());
      scope.add(op.getResult(), result);
    } else if (auto op = dyn_cast<OptimizationBarrierOp>(operation)) {
      auto operand = scope.find(op.getOperand());
      auto results = optimizationBarrierOp(operand);
      scope.add(op.getResults(), results);
    } else if (auto op = dyn_cast<OrOp>(operation)) {
      auto lhs = scope.findTensor(op.getLhs());
      auto rhs = scope.findTensor(op.getRhs());
      auto result = orOp(lhs, rhs, op.getType());
      scope.add(op.getResult(), result);
    } else if (auto op = dyn_cast<OutfeedOp>(operation)) {
      auto inputs = scope.findTensors(op.getInputs());
      auto token = scope.findToken(op.getToken());
      auto result = outfeedOp(inputs, token, process);
      scope.add(op.getResult(), result);
    } else if (auto op = dyn_cast<PadOp>(operation)) {
      auto operand = scope.findTensor(op.getOperand());
      auto paddingValue = scope.findTensor(op.getPaddingValue());
      auto edgePaddingLow = Sizes(op.getEdgePaddingLow());
      auto interiorPadding = Sizes(op.getInteriorPadding());
      auto result = padOp(operand, paddingValue, edgePaddingLow,
                          interiorPadding, op.getType());
      scope.add(op.getResult(), result);
    } else if (auto op = dyn_cast<PartitionIdOp>(operation)) {
      auto result = partitionIdOp(process, op.getContext());
      scope.add(op.getResult(), result);
    } else if (auto op = dyn_cast<PopulationCountOp>(operation)) {
      auto operand = scope.findTensor(op.getOperand());
      auto result = populationCountOp(operand, op.getType());
      scope.add(op.getResult(), result);
    } else if (auto op = dyn_cast<PowOp>(operation)) {
      auto lhs = scope.findTensor(op.getLhs());
      auto rhs = scope.findTensor(op.getRhs());
      auto result = powerOp(lhs, rhs, op.getType());
      scope.add(op.getResult(), result);
    } else if (auto op = dyn_cast<RealOp>(operation)) {
      auto operand = scope.findTensor(op.getOperand());
      auto result = realOp(operand, op.getType());
      scope.add(op.getResult(), result);
    } else if (auto op = dyn_cast<RecvOp>(operation)) {
      auto token = scope.findToken(op.getToken());
      ChannelId channelId = 0;
      if (auto channelHandle = op.getChannelHandle())
        channelId = channelHandle.getHandle();
      auto results = recvOp(token, channelId, process);
      scope.add(op.getResults(), results);
    } else if (auto op = dyn_cast<ReduceOp>(operation)) {
      auto inputs = scope.findTensors(op.getInputs());
      auto initValues = scope.findTensors(op.getInitValues());
      SmallVector<ShapedType> resultTypes;
      for (auto resultType : op.getResultTypes())
        resultTypes.push_back(cast<ShapedType>(resultType));
      auto results = reduceOp(inputs, initValues, Axes(op.getDimensions()),
                              op.getBody(), process, scope, resultTypes);
      scope.add(op.getResults(), results);
    } else if (auto op = dyn_cast<ReducePrecisionOp>(operation)) {
      auto operand = scope.findTensor(op.getOperand());
      int32_t exponentBits = op.getExponentBits();
      int32_t mantissaBits = op.getMantissaBits();
      auto result =
          reducePrecisionOp(operand, exponentBits, mantissaBits, op.getType());
      scope.add(op.getResult(), result);
    } else if (auto op = dyn_cast<ReduceScatterOp>(operation)) {
      auto operand = scope.findTensor(op.getOperand());
      int64_t scatterDimension = op.getScatterDimension();
      auto replicaGroups = getReplicaGroups(op.getReplicaGroups());

      ChannelId channelId = 0;
      if (auto channelHandle = op.getChannelHandle())
        channelId = channelHandle->getHandle();

      auto result =
          reduceScatterOp(operand, scatterDimension, replicaGroups, channelId,
                          op.getUseGlobalDeviceIds(), op.getComputation(),
                          process, scope, op.getType());
      scope.add(op.getResult(), result);
    } else if (auto op = dyn_cast<ReduceWindowOp>(operation)) {
      auto inputs = scope.findTensors(op.getInputs());
      auto initValues = scope.findTensors(op.getInitValues());
      int64_t rank = inputs[0].getRank();

      Sizes windowStrides(rank, 1);
      if (auto windowStridesAttr = op.getWindowStrides())
        windowStrides = Sizes(*windowStridesAttr);

      Sizes baseDilations(rank, 1);
      if (auto baseDilationsAttr = op.getBaseDilations())
        baseDilations = Sizes(*baseDilationsAttr);

      Sizes windowDilations(rank, 1);
      if (auto windowDilationsAttr = op.getWindowDilations())
        windowDilations = Sizes(*windowDilationsAttr);

      Sizes paddingLow(rank, 0), paddingHigh(rank, 0);
      if (auto paddingAttr = op.getPadding()) {
        auto paddingOrErr = hlo::convertPaddingAttribute(paddingAttr, {});
        if (failed(paddingOrErr))
          report_fatal_error(invalidArgument("Invalid padding format found."));
        for (auto i = 0; i < static_cast<int64_t>(paddingOrErr->size()); ++i) {
          paddingLow[i] = (*paddingOrErr)[i].first;
          paddingHigh[i] = (*paddingOrErr)[i].second;
        }
      }

      SmallVector<ShapedType> resultTypes;
      for (auto resultType : op.getResultTypes())
        resultTypes.push_back(cast<ShapedType>(resultType));

      auto results = reduceWindowOp(
          inputs, initValues, Sizes(op.getWindowDimensions()), windowStrides,
          baseDilations, windowDilations, paddingLow, paddingHigh, op.getBody(),
          process, scope, resultTypes);
      scope.add(op.getResults(), results);
    } else if (auto op = dyn_cast<RemOp>(operation)) {
      auto lhs = scope.findTensor(op.getLhs());
      auto rhs = scope.findTensor(op.getRhs());
      auto result = remOp(lhs, rhs, op.getType());
      scope.add(op.getResult(), result);
    } else if (auto op = dyn_cast<ReplicaIdOp>(operation)) {
      auto result = replicaIdOp(process, op.getContext());
      scope.add(op.getResult(), result);
    } else if (auto op = dyn_cast<ReshapeOp>(operation)) {
      auto operand = scope.findTensor(op.getOperand());
      auto result = reshapeOp(operand, op.getType());
      scope.add(op.getResult(), result);
    } else if (auto op = dyn_cast<func::ReturnOp>(operation)) {
      return scope.find(op.getOperands());
    } else if (auto op = dyn_cast<ReturnOp>(operation)) {
      return scope.find(op.getResults());
    } else if (auto op = dyn_cast<ReverseOp>(operation)) {
      auto operand = scope.findTensor(op.getOperand());
      auto dimensions = Axes(op.getDimensions());
      auto result = reverseOp(operand, dimensions, op.getType());
      scope.add(op.getResult(), result);
    } else if (isa<RngBitGeneratorOp>(operation)) {
      failOnDecomposableOp(operation);
    } else if (isa<RngOp>(operation)) {
      failOnDecomposableOp(operation);
    } else if (auto op = dyn_cast<RoundNearestEvenOp>(operation)) {
      auto operand = scope.findTensor(op.getOperand());
      auto result = roundNearestEvenOp(operand, op.getType());
      scope.add(op.getResult(), result);
    } else if (auto op = dyn_cast<RoundOp>(operation)) {
      auto operand = scope.findTensor(op.getOperand());
      auto result = roundOp(operand, op.getType());
      scope.add(op.getResult(), result);
    } else if (auto op = dyn_cast<RsqrtOp>(operation)) {
      auto operand = scope.findTensor(op.getOperand());
      auto result = rsqrtOp(operand, op.getType());
      scope.add(op.getResult(), result);
    } else if (auto op = dyn_cast<ScatterOp>(operation)) {
      auto inputs = scope.findTensors(op.getInputs());
      auto scatterIndices = scope.findTensor(op.getScatterIndices());
      auto updates = scope.findTensors(op.getUpdates());
      auto scatterDimensionNumbers = op.getScatterDimensionNumbers();
      Axes updateWindowDims(scatterDimensionNumbers.getUpdateWindowDims());
      Axes insertedWindowDims(scatterDimensionNumbers.getInsertedWindowDims());
      Axes inputBatchingDims(scatterDimensionNumbers.getInputBatchingDims());
      Axes scatterIndicesBatchingDims(
          scatterDimensionNumbers.getScatterIndicesBatchingDims());
      Axes scatterDimsToOperandDims(
          scatterDimensionNumbers.getScatterDimsToOperandDims());
      Axis indexVectorDim(scatterDimensionNumbers.getIndexVectorDim());
      auto &updateComputation = op.getUpdateComputation();
      SmallVector<ShapedType> resultTypes(op->getResultTypes());
      auto results = scatterOp(inputs, scatterIndices, updates,
                               updateWindowDims, insertedWindowDims,
                               inputBatchingDims, scatterIndicesBatchingDims,
                               scatterDimsToOperandDims, indexVectorDim,
                               updateComputation, process, scope, resultTypes);
      scope.add(op.getResults(), results);
    } else if (auto op = dyn_cast<SelectAndScatterOp>(operation)) {
      auto operand = scope.findTensor(op.getOperand());
      auto source = scope.findTensor(op.getSource());
      auto initValue = scope.findTensor(op.getInitValue());
      auto rank = operand.getRank();

      Sizes windowDimensions(rank, 1);
      if (auto windowDimensionsAttr = op.getWindowDimensions())
        windowDimensions.assign(windowDimensionsAttr->begin(),
                                windowDimensionsAttr->end());

      Sizes windowStrides(rank, 1);
      if (auto windowStridesAttr = op.getWindowStrides())
        windowStrides.assign(windowStridesAttr->begin(),
                             windowStridesAttr->end());

      Sizes paddingLow(rank, 0);
      if (auto padding = op.getPadding()) {
        auto paddingOrErr = hlo::convertPaddingAttribute(padding, {});
        if (failed(paddingOrErr))
          report_fatal_error(invalidArgument("Invalid padding format found."));
        for (auto i = 0; i < static_cast<int64_t>(paddingOrErr->size()); ++i) {
          paddingLow[i] = (*paddingOrErr)[i].first;
        }
      }

      auto result =
          selectAndScatterOp(operand, source, initValue, windowDimensions,
                             windowStrides, paddingLow, op.getSelect(),
                             op.getScatter(), process, scope, op.getType());
      scope.add(op.getResult(), result);
    } else if (auto op = dyn_cast<SelectOp>(operation)) {
      auto pred = scope.findTensor(op.getPred());
      auto onTrue = scope.findTensor(op.getOnTrue());
      auto onFalse = scope.findTensor(op.getOnFalse());
      auto result = selectOp(pred, onTrue, onFalse, op.getType());
      scope.add(op.getResult(), result);
    } else if (auto op = dyn_cast<SendOp>(operation)) {
      auto inputs = scope.findTensors(op.getInputs());
      auto token = scope.findToken(op.getToken());
      ChannelId channelId = 0;
      if (auto channelHandle = op.getChannelHandle())
        channelId = channelHandle.getHandle();
      auto result = sendOp(inputs, token, channelId, process);
      scope.add(op.getResult(), result);
    } else if (auto op = dyn_cast<ShiftLeftOp>(operation)) {
      auto lhs = scope.findTensor(op.getLhs());
      auto rhs = scope.findTensor(op.getRhs());
      auto result = shiftLeftOp(lhs, rhs, op.getType());
      scope.add(op.getResult(), result);
    } else if (auto op = dyn_cast<ShiftRightArithmeticOp>(operation)) {
      auto lhs = scope.findTensor(op.getLhs());
      auto rhs = scope.findTensor(op.getRhs());
      auto result = shiftRightArithmeticOp(lhs, rhs, op.getType());
      scope.add(op.getResult(), result);
    } else if (auto op = dyn_cast<ShiftRightLogicalOp>(operation)) {
      auto lhs = scope.findTensor(op.getLhs());
      auto rhs = scope.findTensor(op.getRhs());
      auto result = shiftRightLogicalOp(lhs, rhs, op.getType());
      scope.add(op.getResult(), result);
    } else if (auto op = dyn_cast<SignOp>(operation)) {
      auto operand = scope.findTensor(op.getOperand());
      auto result = signOp(operand, op.getType());
      scope.add(op.getResult(), result);
    } else if (auto op = dyn_cast<SineOp>(operation)) {
      auto operand = scope.findTensor(op.getOperand());
      auto result = sineOp(operand, op.getType());
      scope.add(op.getResult(), result);
    } else if (auto op = dyn_cast<SliceOp>(operation)) {
      auto operand = scope.findTensor(op.getOperand());
      auto startIndices = Sizes(op.getStartIndices());
      auto strides = Sizes(op.getStrides());
      auto result = sliceOp(operand, startIndices, strides, op.getType());
      scope.add(op.getResult(), result);
    } else if (auto op = dyn_cast<SortOp>(operation)) {
      auto operands = scope.findTensors(op.getInputs());
      auto dimension = op.getDimension();
      auto isStable = op.getIsStable();
      auto &comparator = op.getComparator();
      auto results =
          sortOp(operands, dimension, isStable, comparator, process, scope);
      scope.add(op.getResults(), results);
    } else if (auto op = dyn_cast<SqrtOp>(operation)) {
      auto operand = scope.findTensor(op.getOperand());
      auto result = sqrtOp(operand, op.getType());
      scope.add(op.getResult(), result);
    } else if (auto op = dyn_cast<SubtractOp>(operation)) {
      auto lhs = scope.findTensor(op.getLhs());
      auto rhs = scope.findTensor(op.getRhs());
      auto result = subtractOp(lhs, rhs, op.getType());
      scope.add(op.getResult(), result);
    } else if (auto op = dyn_cast<TanOp>(operation)) {
      auto operand = scope.findTensor(op.getOperand());
      auto result = tanOp(operand, op.getType());
      scope.add(op.getResult(), result);
    } else if (auto op = dyn_cast<TanhOp>(operation)) {
      auto operand = scope.findTensor(op.getOperand());
      auto result = tanhOp(operand, op.getType());
      scope.add(op.getResult(), result);
    } else if (isa<TorchIndexSelectOp>(operation)) {
      failOnDecomposableOp(operation);
    } else if (auto op = dyn_cast<TransposeOp>(operation)) {
      auto operand = scope.findTensor(op.getOperand());
      auto permutation = Axes(op.getPermutation());
      auto result = transposeOp(operand, permutation, op.getType());
      scope.add(op.getResult(), result);
    } else if (isa<TriangularSolveOp>(operation)) {
      failOnDecomposableOp(operation);
    } else if (auto op = dyn_cast<TupleOp>(operation)) {
      auto val = scope.find(op.getVal());
      auto result = tupleOp(val, cast<TupleType>(op.getType()));
      scope.add(op.getResult(), result);
    } else if (isa<UnaryEinsumOp>(operation)) {
      failOnDecomposableOp(operation);
    } else if (auto op = dyn_cast<WhileOp>(operation)) {
      auto operand = scope.find(op.getOperand());
      auto &cond = op.getCond();
      auto &body = op.getBody();
      auto results = whileOp(operand, cond, body, fallback, process, scope);
      scope.add(op.getResults(), results);
    } else if (auto op = dyn_cast<XorOp>(operation)) {
      auto lhs = scope.findTensor(op.getLhs());
      auto rhs = scope.findTensor(op.getRhs());
      auto result = xorOp(lhs, rhs, op.getType());
      scope.add(op.getResult(), result);
    } else {
      if (!fallback)
        report_fatal_error(invalidArgument("Unsupported op: %s",
                                           debugString(operation).c_str()));
      auto status = (*fallback)(operation, scope, process);
      if (status) llvm::report_fatal_error(std::move(status));
    }
  }

  llvm::report_fatal_error("Expected a terminator when evaluating a region");
}

Tensor absOp(const Tensor &operand, ShapedType resultType) {
  Tensor result(resultType);
  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it, abs(operand.get(*it)));
  return result;
}

Tensor addOp(const Tensor &lhs, const Tensor &rhs, ShapedType resultType) {
  Tensor result(resultType);
  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it, lhs.get(*it) + rhs.get(*it));
  return result;
}

Token afterAllOp(ArrayRef<Token> inputs, MLIRContext *context) {
  return Token(context);
}

SmallVector<InterpreterValue> allGatherOp(
    ArrayRef<Tensor> operands, int64_t allGatherDim,
    SmallVector<SmallVector<uint32_t>> replicaGroups, ChannelId channelId,
    bool useGlobalDeviceIds, Process *process,
    ArrayRef<ShapedType> resultTypes) {
  if (!process)
    llvm::report_fatal_error(
        "all_gather is only supported when run via interpreter.run_parallel");

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
  auto rendezvousResult =
      process->rendezvous(*processGroup, channelId, operands);

  SmallVector<InterpreterValue> results(resultTypes.size());
  for (const auto &[resultIndex, resultType] : llvm::enumerate(resultTypes)) {
    auto operandIndex = resultIndex;
    auto operandsAtIndex =
        llvm::map_to_vector(*processGroup, [&](const ProcessId &id) {
          return (rendezvousResult.lookup(id))[operandIndex];
        });
    results[resultIndex] =
        concatenateOp(operandsAtIndex, allGatherDim, resultType);
  }
  return results;
}

SmallVector<InterpreterValue> allReduceOp(
    ArrayRef<Tensor> operands, SmallVector<SmallVector<uint32_t>> replicaGroups,
    ChannelId channelId, bool useGlobalDeviceIds, Region &computation,
    Process *process, Scope &scope, ArrayRef<ShapedType> resultTypes) {
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
  auto groupOperands = process->rendezvous(*processGroup, channelId, operands)
                           .getSortedTensors();

  SmallVector<InterpreterValue> results(resultTypes.size());
  for (const auto &[resultIndex, resultType] : llvm::enumerate(resultTypes)) {
    Tensor result(resultType);
    for (auto elementIndex = result.index_begin();
         elementIndex != result.index_end(); ++elementIndex) {
      Tensor resultElement;
      for (const auto &processOperands : groupOperands) {
        auto OperandElement =
            constant(processOperands[resultIndex].get(*elementIndex));
        if (resultElement)
          resultElement = eval(computation, {resultElement, OperandElement},
                               /*fallback=*/nullptr, process, &scope)[0]
                              .getTensor();
        else
          resultElement = OperandElement;
      }
      result.set(*elementIndex, resultElement.get({}));
    }
    results[resultIndex] = result;
  }
  return results;
}

SmallVector<InterpreterValue> allToAllOp(
    ArrayRef<Tensor> operands, Axis splitDimension, Axis concatDimension,
    int64_t splitCount, SmallVector<SmallVector<uint32_t>> replicaGroups,
    ChannelId channelId, Process *process, ArrayRef<ShapedType> resultTypes) {
  if (!process)
    llvm::report_fatal_error(
        "all_to_all is only supported when run via interpreter.run_parallel");

  ProcessGroups processGroups;
  if (channelId <= 0) processGroups = process->crossReplica(replicaGroups);
  if (channelId > 0) processGroups = process->crossPartition(replicaGroups);

  auto processGroup = processGroups.findGroup(process->getId());
  if (!processGroup.has_value())
    llvm::report_fatal_error(invalidArgument(
        "Failed to find process group with process_id: (%d, %d)",
        process->getId().replicaId, process->getId().partitionId));
  auto groupOperands =
      process->rendezvous(processGroup.value(), channelId, operands);

  auto receiverIndex = llvm::find(processGroup.value(), process->getId()) -
                       processGroup->begin();
  SmallVector<InterpreterValue> results(resultTypes.size());
  for (const auto &[resultIndex, resultType] : llvm::enumerate(resultTypes)) {
    SmallVector<Tensor> scatteredParts;
    for (const auto &sender : processGroup.value()) {
      auto splitParts =
          split(groupOperands.lookup(sender)[resultIndex], splitCount,
                splitDimension, operands[resultIndex].getType().getContext());
      scatteredParts.push_back(splitParts[receiverIndex]);
    }
    results[resultIndex] =
        concatenateOp(scatteredParts, concatDimension, resultType);
  }
  return results;
}
Tensor andOp(const Tensor &lhs, const Tensor &rhs, ShapedType resultType) {
  Tensor result(resultType);
  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it, lhs.get(*it) & rhs.get(*it));
  return result;
}

Tensor atan2Op(const Tensor &lhs, const Tensor &rhs, ShapedType resultType) {
  Tensor result(resultType);
  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it, atan2(lhs.get(*it), rhs.get(*it)));
  return result;
}

Tensor bitcastConvertOp(const Tensor &operand, ShapedType resultType) {
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

Tensor broadcastInDimOp(const Tensor &operand, const Axes &broadcastDimensions,
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

SmallVector<InterpreterValue> caseOp(const Tensor &index, RegionRange branches,
                                     Process *process, Scope &scope) {
  int64_t indexValue = index.get({}).getIntegerValue().getSExtValue();
  if (indexValue < 0 || indexValue >= static_cast<int64_t>(branches.size()))
    indexValue = branches.size() - 1;

  return eval(*branches[indexValue], {}, /*fallback=*/nullptr, process, &scope);
}

Tensor cbrtOp(const Tensor &operand, ShapedType resultType) {
  Tensor result(resultType);
  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it, cbrt(operand.get(*it)));
  return result;
}

Tensor ceilOp(const Tensor &operand, ShapedType resultType) {
  Tensor result(resultType);
  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it, ceil(operand.get(*it)));
  return result;
}

Tensor clampOp(const Tensor &min, const Tensor &operand, const Tensor &max,
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

Tensor clzOp(const Tensor &operand, ShapedType resultType) {
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

Tensor collectiveBroadcastOp(const Tensor &operand,
                             SmallVector<SmallVector<uint32_t>> replicaGroups,
                             ChannelId channelId, Process *process) {
  if (!process)
    llvm::report_fatal_error(
        "collective_broadcast is only supported when run via "
        "interpreter.run_parallel");

  ProcessGroups processGroups;
  if (channelId <= 0) processGroups = process->crossReplica(replicaGroups);
  if (channelId > 0) processGroups = process->crossPartition(replicaGroups);

  auto processGroup = processGroups.findGroup(process->getId());
  if (processGroup) {
    return process->rendezvous(*processGroup, channelId, {operand})
        .lookup((*processGroup)[0])
        .front();
  }
  return broadcastInDimOp(constant(0.0, operand.getElementType()), {},
                          operand.getType());
}

Tensor collectivePermuteOp(const Tensor &operand,
                           SmallVector<SmallVector<uint32_t>> sourceTargetPairs,
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
        process->rendezvous(processGroup, channelId, {operand});
    if (to != process->getId()) continue;
    result = rendezvousResult.lookup(from).front();
  }

  if (result) return result;
  return broadcastInDimOp(constant(0.0, operand.getElementType()), {},
                          operand.getType());
}

Tensor compareOp(const Tensor &lhs, const Tensor &rhs,
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

Tensor complexOp(const Tensor &lhs, const Tensor &rhs, ShapedType resultType) {
  Tensor result(resultType);
  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it, complex(lhs.get(*it), rhs.get(*it)));
  return result;
}

Tensor concatenateOp(ArrayRef<Tensor> inputs, Axis dimension,
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

Tensor constantOp(ElementsAttr value) {
  return makeTensor(cast<DenseElementsAttr>(value));
}

Tensor convertOp(const Tensor &operand, ShapedType resultType) {
  Tensor result(resultType);
  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it, convert(result.getElementType(), operand.get(*it)));
  return result;
}

Tensor convolutionOp(
    const Tensor &lhs, const Tensor &rhs, ArrayRef<int64_t> windowStrides,
    ArrayRef<std::pair<int64_t, int64_t>> padding,
    ArrayRef<int64_t> lhsDilation, ArrayRef<int64_t> rhsDilation,
    ArrayRef<bool> windowReversal, Axis inputBatchDimension,
    Axis inputFeatureDimension, const Axes &inputSpatialDimensions,
    Axis kernelInputFeatureDimension, Axis kernelOutputFeatureDimension,
    const Axes &kernelSpatialDimensions, Axis outputBatchDimension,
    Axis outputFeatureDimension, const Axes &outputSpatialDimensions,
    int64_t featureGroupCount, int64_t batchGroupCount, ShapedType resultType) {
  Tensor result(resultType);

  if (featureGroupCount > 1) {
    auto lhses = split(lhs, featureGroupCount, inputFeatureDimension,
                       resultType.getContext());
    auto rhses = split(rhs, featureGroupCount, kernelOutputFeatureDimension,
                       resultType.getContext());
    SmallVector<Tensor> results;
    for (auto [left, right] : llvm::zip(lhses, rhses))
      results.push_back(convolutionOp(
          left, right, windowStrides, padding, lhsDilation, rhsDilation,
          windowReversal, inputBatchDimension, inputFeatureDimension,
          inputSpatialDimensions, kernelInputFeatureDimension,
          kernelOutputFeatureDimension, kernelSpatialDimensions,
          outputBatchDimension, outputFeatureDimension, outputSpatialDimensions,
          /*featureGroupCount=*/1, batchGroupCount, /*precisionConfig=*/{},
          resultType));

    return concatenateOp(results, outputFeatureDimension, result.getType());
  }

  if (batchGroupCount > 1) {
    auto lhses = split(lhs, batchGroupCount, inputBatchDimension,
                       resultType.getContext());
    auto rhses = split(rhs, batchGroupCount, kernelOutputFeatureDimension,
                       resultType.getContext());
    SmallVector<Tensor> results;
    for (auto [left, right] : llvm::zip(lhses, rhses))
      results.push_back(convolutionOp(
          left, right, windowStrides, padding, lhsDilation, rhsDilation,
          windowReversal, inputBatchDimension, inputFeatureDimension,
          inputSpatialDimensions, kernelInputFeatureDimension,
          kernelOutputFeatureDimension, kernelSpatialDimensions,
          outputBatchDimension, outputFeatureDimension, outputSpatialDimensions,
          featureGroupCount, /*batchGroupCount=*/1, /*precisionConfig=*/{},
          resultType));

    return concatenateOp(results, outputFeatureDimension, result.getType());
  }

  Axes lhsPermutation;
  lhsPermutation.push_back(inputBatchDimension);
  lhsPermutation.append(inputSpatialDimensions.begin(),
                        inputSpatialDimensions.end());
  lhsPermutation.push_back(inputFeatureDimension);

  auto lhsWindowDimensions = concatAndPermute<int64_t>(
      lhs.getShape()[inputBatchDimension],
      extractElements(rhs.getShape(), kernelSpatialDimensions),
      lhs.getShape()[inputFeatureDimension], lhsPermutation);

  auto lhsWindowStrides = concatAndPermute<int64_t>(
      1L, llvm::to_vector(windowStrides), 1L, lhsPermutation);

  auto lhsBaseDilations =
      concatAndPermute<int64_t>(0L, Sizes(lhsDilation) - 1, 0L, lhsPermutation);

  auto lhsWindowDilations = concatAndPermute<int64_t>(
      1L, llvm::to_vector(rhsDilation), 1L, lhsPermutation);

  Sizes lhsPaddingLow, lhsPaddingHigh;
  for (auto paddingPair : concatAndPermute<std::pair<int64_t, int64_t>>(
           {0, 0}, llvm::to_vector(padding), {0, 0}, lhsPermutation)) {
    lhsPaddingLow.push_back(paddingPair.first);
    lhsPaddingHigh.push_back(paddingPair.second);
  }

  auto paddingValue = constant(0.0, result.getElementType());
  auto paddedLhs = padOp(lhs, paddingValue, lhsPaddingLow, lhsPaddingHigh,
                         Sizes(lhsBaseDilations));

  IndexSpaceIterator outputSpatialIndexIt(
      extractElements(result.getShape(), outputSpatialDimensions),
      Index(outputSpatialDimensions.size()));
  IndexSpaceIterator outputSpatialIndexItEnd(
      extractElements(result.getShape(), outputSpatialDimensions));
  for (; outputSpatialIndexIt != outputSpatialIndexItEnd;
       ++outputSpatialIndexIt) {
    Sizes lhsWindowStart;
    for (auto [i, offset] : llvm::enumerate(concatAndPermute<int64_t>(
             0L, *outputSpatialIndexIt, 0L, lhsPermutation)))
      lhsWindowStart.push_back(lhsWindowStrides[i] * offset);

    Sizes limitIndices;
    for (size_t i = 0; i < lhsWindowStart.size(); ++i)
      limitIndices.push_back(std::min(
          lhsWindowStart[i] + lhsWindowDimensions[i] * lhsWindowDilations[i],
          paddedLhs.getShape()[i]));

    auto lhsWindow = sliceOp(paddedLhs, lhsWindowStart, limitIndices,
                             Sizes(lhsWindowDilations));

    Axes reverseDims;
    for (auto [i, isReverse] : llvm::enumerate(windowReversal))
      if (isReverse) reverseDims.push_back(inputSpatialDimensions[i]);
    auto reversedLhsWindow =
        reverseOp(lhsWindow, reverseDims, lhsWindow.getType());

    Axes lhsContractingDimensions(inputSpatialDimensions);
    lhsContractingDimensions.push_back(inputFeatureDimension);

    Axes rhsContractingDimensions(kernelSpatialDimensions);
    rhsContractingDimensions.push_back(kernelInputFeatureDimension);

    auto dotProduct =
        dotGeneralOp(reversedLhsWindow, rhs, lhsContractingDimensions,
                     rhsContractingDimensions);

    Sizes resultNonSpatialDims;
    for (auto i = 0; i < result.getRank(); ++i)
      if (llvm::find(outputSpatialDimensions, i) ==
          outputSpatialDimensions.end())
        resultNonSpatialDims.push_back(result.getShape()[i]);

    Axes resultPermutation;
    resultPermutation.push_back(outputBatchDimension);
    resultPermutation.append(outputSpatialDimensions.begin(),
                             outputSpatialDimensions.end());
    resultPermutation.push_back(outputFeatureDimension);

    IndexSpaceIterator resultNonSpatialIt(resultNonSpatialDims,
                                          Index(resultNonSpatialDims.size()));
    for (auto dotProductIt = dotProduct.index_begin();
         dotProductIt != dotProduct.index_end();
         ++dotProductIt, ++resultNonSpatialIt) {
      Index resultIndex(concatAndPermute<int64_t>(
          (*resultNonSpatialIt)[0], *outputSpatialIndexIt,
          (*resultNonSpatialIt)[1], resultPermutation));
      result.set(resultIndex, dotProduct.get(*dotProductIt));
    }
  }
  return result;
}

Tensor cosineOp(const Tensor &operand, ShapedType resultType) {
  Tensor result(resultType);
  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it, cosine(operand.get(*it)));
  return result;
}

Tensor divideOp(const Tensor &lhs, const Tensor &rhs, ShapedType resultType) {
  Tensor result(resultType);
  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it, lhs.get(*it) / rhs.get(*it));
  return result;
}

Tensor dotGeneralOp(const Tensor &lhs, const Tensor &rhs,
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
    auto resultElement = getZeroValueOfType(resultType.getElementType());
    while (true) {
      resultElement =
          resultElement +
          convert(resultType.getElementType(), lhs.get(lhsIndex)) *
              convert(resultType.getElementType(), rhs.get(rhsIndex));
      if (failed(incrementIndices())) break;
    }
    result.set(resultIndex, resultElement);
  }
  return result;
}

Tensor dynamicSliceOp(const Tensor &operand, ArrayRef<Tensor> startIndices,
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

Tensor dynamicUpdateSliceOp(const Tensor &operand, const Tensor &update,
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

Tensor expm1Op(const Tensor &operand, ShapedType resultType) {
  Tensor result(resultType);
  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it, exponentialMinusOne(operand.get(*it)));
  return result;
}

Tensor exponentialOp(const Tensor &operand, ShapedType resultType) {
  Tensor result(resultType);
  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it, exponential(operand.get(*it)));
  return result;
}

Tensor floorOp(const Tensor &operand, ShapedType resultType) {
  Tensor result(resultType);
  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it, floor(operand.get(*it)));
  return result;
}

Tensor gatherOp(const Tensor &operand, const Tensor &startIndices,
                const Axes &offsetDims, const Axes &collapsedSliceDims,
                const Axes &operandBatchingDims,
                const Axes &startIndicesBatchingDims, const Axes &startIndexMap,
                Axis indexVectorDim, const Sizes &sliceSizes,
                bool indicesAreSorted, ShapedType resultType) {
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
    auto startIndex = evalIndex(sliceOp(startIndices, startIndicesIndex));

    Index fullStartIndex(operand.getRank(), 0);
    for (auto dOperand : operand.getAxes()) {
      auto dStartIt = llvm::find(startIndexMap, dOperand);
      if (dStartIt == startIndexMap.end()) continue;
      auto dStart = dStartIt - startIndexMap.begin();
      fullStartIndex[dOperand] = std::clamp<int64_t>(
          startIndex[dStart], 0ll,
          operand.getShape()[dOperand] - sliceSizes[dOperand]);
    }

    Index fullBatchingIndex(operand.getRank(), 0);
    for (auto dOperand : operand.getAxes()) {
      auto dBatchingIt = llvm::find(operandBatchingDims, dOperand);
      if (dBatchingIt == operandBatchingDims.end()) continue;
      auto iBatching = dBatchingIt - operandBatchingDims.begin();
      auto dStart = startIndicesBatchingDims[iBatching];
      fullBatchingIndex[dOperand] = startIndicesIndex[dStart];
    }

    Index offsetIndex;
    for (auto d : offsetDims) offsetIndex.push_back(resultIndex[d]);

    Index fullOffsetIndex(operand.getRank(), 0);
    for (size_t i = 0, oi = 0; i < fullOffsetIndex.size(); ++i) {
      if (llvm::is_contained(collapsedSliceDims, i) ||
          llvm::is_contained(operandBatchingDims, i))
        continue;
      fullOffsetIndex[i] = offsetIndex[oi++];
    }

    auto operandIndex = fullStartIndex + fullBatchingIndex + fullOffsetIndex;
    result.set(resultIndex, operand.get(operandIndex));
  }
  return result;
}

Tensor getDimensionSizeOp(const Tensor &operand, Axis dimension,
                          ShapedType resultType) {
  Tensor result(resultType);
  result.set(
      {}, convert(resultType.getElementType(), operand.getShape()[dimension]));
  return result;
}

InterpreterValue getTupleElementOp(const Tuple &operand, int32_t index) {
  return operand.get(index);
}

SmallVector<InterpreterValue> ifOp(const Tensor &pred, Region &trueBranch,
                                   Region &falseBranch, Process *process,
                                   Scope &scope) {
  return pred.get({}).getBooleanValue()
             ? eval(trueBranch, {}, /*fallback=*/nullptr, process, &scope)
             : eval(falseBranch, {}, /*fallback=*/nullptr, process, &scope);
}

Tensor imagOp(const Tensor &operand, ShapedType resultType) {
  Tensor result(resultType);
  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it, imag(operand.get(*it)));
  return result;
}

SmallVector<InterpreterValue> infeedOp(Token token, Process *process,
                                       Region &region, Scope &scope) {
  if (!process)
    llvm::report_fatal_error(
        "infeed is only supported when run via interpreter.run_parallel");

  auto mnemonic = process->infeed();
  auto results = eval(region.getParentOfType<ModuleOp>()
                          .lookupSymbol<func::FuncOp>(mnemonic)
                          .getBody(),
                      {}, /*fallback=*/nullptr, process, &scope);
  results.push_back(token);
  return results;
}

Tensor iotaOp(Axis iotaDimension, ShapedType resultType) {
  Tensor result(resultType);
  auto elementType = result.getElementType();
  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it, convert(elementType, (*it)[iotaDimension]));
  return result;
}

Tensor isFiniteOp(const Tensor &operand, ShapedType resultType) {
  Tensor result(resultType);
  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it, isFinite(operand.get(*it)));
  return result;
}

Tensor log1pOp(const Tensor &operand, ShapedType resultType) {
  Tensor result(resultType);
  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it, logPlusOne(operand.get(*it)));
  return result;
}

Tensor logOp(const Tensor &operand, ShapedType resultType) {
  Tensor result(resultType);
  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it, log(operand.get(*it)));
  return result;
}

Tensor logisticOp(const Tensor &operand, ShapedType resultType) {
  Tensor result(resultType);
  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it, logistic(operand.get(*it)));
  return result;
}

Tensor mapOp(ArrayRef<Tensor> inputs, Region &computation, Process *process,
             Scope &scope, ShapedType resultType) {
  Tensor result(resultType);
  for (auto it = result.index_begin(); it != result.index_end(); ++it) {
    SmallVector<InterpreterValue> args;
    for (size_t i = 0; i < inputs.size(); ++i) {
      Tensor tensor(cast<ShapedType>(computation.getArgument(i).getType()));
      tensor.set({}, inputs[i].get(*it));
      args.emplace_back(tensor);
    }
    result.set(*it,
               eval(computation, args, /*fallback=*/nullptr, process, &scope)[0]
                   .getTensor()
                   .get({}));
  }
  return result;
}

Tensor maxOp(const Tensor &lhs, const Tensor &rhs, ShapedType resultType) {
  Tensor result(resultType);
  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it, max(lhs.get(*it), rhs.get(*it)));
  return result;
}

Tensor minOp(const Tensor &lhs, const Tensor &rhs, ShapedType resultType) {
  Tensor result(resultType);
  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it, min(lhs.get(*it), rhs.get(*it)));
  return result;
}

Tensor multiplyOp(const Tensor &lhs, const Tensor &rhs, ShapedType resultType) {
  Tensor result(resultType);
  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it, lhs.get(*it) * rhs.get(*it));
  return result;
}

Tensor negOp(const Tensor &operand, ShapedType resultType) {
  Tensor result(resultType);
  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it, -operand.get(*it));
  return result;
}

Tensor notOp(const Tensor &operand, ShapedType resultType) {
  Tensor result(resultType);
  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it, ~operand.get(*it));
  return result;
}

SmallVector<InterpreterValue> optimizationBarrierOp(
    ArrayRef<InterpreterValue> operand) {
  return SmallVector<InterpreterValue>(operand);
}

Tensor orOp(const Tensor &lhs, const Tensor &rhs, ShapedType resultType) {
  Tensor result(resultType);
  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it, lhs.get(*it) | rhs.get(*it));
  return result;
}

Token outfeedOp(ArrayRef<Tensor> inputs, Token token, Process *process) {
  if (!process)
    llvm::report_fatal_error(
        "outfeed is only supported when run via interpreter.run_parallel");

  process->outfeed(inputs);
  return token;
}

Tensor padOp(const Tensor &operand, const Tensor &paddingValue,
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

Tensor partitionIdOp(Process *process, MLIRContext *context) {
  if (!process)
    llvm::report_fatal_error(
        "partition_id is only supported when run via interpreter.run_parallel");
  auto partitionId = process->getId().partitionId;
  auto elementType = IntegerType::get(context, 32, IntegerType::Unsigned);
  return constant(APInt(32, partitionId), elementType);
}

Tensor populationCountOp(const Tensor &operand, ShapedType resultType) {
  Tensor result(resultType);
  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it, popcnt(operand.get(*it)));
  return result;
}

Tensor powerOp(const Tensor &lhs, const Tensor &rhs, ShapedType resultType) {
  Tensor result(resultType);
  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it, power(lhs.get(*it), rhs.get(*it)));
  return result;
}

Tensor realOp(const Tensor &operand, ShapedType resultType) {
  Tensor result(resultType);
  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it, real(operand.get(*it)));
  return result;
}

SmallVector<InterpreterValue> recvOp(Token token, ChannelId channelId,
                                     Process *process) {
  SmallVector<InterpreterValue> results;
  for (const auto &tensor : process->recv(channelId)) results.push_back(tensor);
  results.push_back(token);
  return results;
}

SmallVector<Tensor> reduceOp(ArrayRef<Tensor> inputs,
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

    auto bodyResult =
        eval(body, bodyArgs, /*fallback=*/nullptr, process, &scope);
    for (auto [result, value] : llvm::zip(results, bodyResult))
      result.set(resultIndex, value.getTensor().get({}));
  }
  return results;
}

Tensor reducePrecisionOp(const Tensor &operand, int32_t exponentBits,
                         int32_t mantissaBits, ShapedType resultType) {
  Tensor result(resultType);
  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it,
               reducePrecision(operand.get(*it), exponentBits, mantissaBits));
  return result;
}

Tensor reduceScatterOp(const Tensor &operand, int64_t scatterDimension,
                       SmallVector<SmallVector<uint32_t>> replicaGroups,
                       ChannelId channelId, bool useGlobalDeviceIds,
                       Region &region, Process *process, Scope &scope,
                       ShapedType returnType) {
  if (!process)
    llvm::report_fatal_error(
        "reduce_scatter is only supported when run via "
        "interpreter.run_parallel");

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

  auto reducedValue =
      allReduceOp(operand, replicaGroups, channelId, useGlobalDeviceIds, region,
                  process, scope, operand.getType());

  auto parts = split(reducedValue.front().getTensor(), processGroups[0].size(),
                     scatterDimension, operand.getType().getContext());

  Tensor result(returnType);
  for (auto [receiverIndex, sender] : llvm::enumerate(*processGroup)) {
    if (sender == process->getId()) {
      result = parts[receiverIndex];
      break;
    }
  }

  return result;
}

SmallVector<Tensor> reduceWindowOp(
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
    paddedInputs.push_back(
        padOp(input, initValue, paddingLow, paddingHigh, baseDilations - 1));
  for (auto resultIt = results[0].index_begin();
       resultIt != results[0].index_end(); ++resultIt) {
    SmallVector<Tensor> windows;
    auto windowStart = (*resultIt) * windowStrides;
    auto windowEnd = windowStart + (windowDimensions - 1) * windowDilations + 1;
    for (const auto &paddedInput : paddedInputs)
      windows.push_back(
          sliceOp(paddedInput, windowStart, windowEnd, windowDilations));

    auto reducedValues = reduceOp(windows, initValues, inputs[0].getAxes(),
                                  body, process, scope);
    for (auto [result, value] : llvm::zip(results, reducedValues))
      result.set(*resultIt, value.get({}));
  }
  return results;
}

Tensor remOp(const Tensor &lhs, const Tensor &rhs, ShapedType resultType) {
  Tensor result(resultType);
  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it, rem(lhs.get(*it), rhs.get(*it)));
  return result;
}

Tensor replicaIdOp(Process *process, MLIRContext *context) {
  if (!process)
    llvm::report_fatal_error(
        "replica_id is only supported when run via interpreter.run_parallel");
  auto replicaId = process->getId().replicaId;
  auto elementType = IntegerType::get(context, 32, IntegerType::Unsigned);
  return constant(APInt(32, replicaId), elementType);
}

Tensor reshapeOp(const Tensor &operand, ShapedType resultType) {
  Tensor result(resultType);
  for (auto resultIt = result.index_begin(), operandIt = operand.index_begin();
       resultIt != result.index_end(); ++resultIt, ++operandIt) {
    auto resultIndex = *resultIt;
    auto operandIndex = *operandIt;
    result.set(resultIndex, operand.get(operandIndex));
  }
  return result;
}

Tensor reverseOp(const Tensor &operand, const Axes &dimensions,
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

Tensor roundNearestEvenOp(const Tensor &operand, ShapedType resultType) {
  Tensor result(resultType);
  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it, roundNearestEven(operand.get(*it)));
  return result;
}

Tensor roundOp(const Tensor &operand, ShapedType resultType) {
  Tensor result(resultType);
  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it, roundNearestAfz(operand.get(*it)));
  return result;
}

Tensor rsqrtOp(const Tensor &operand, ShapedType resultType) {
  Tensor result(resultType);
  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it, rsqrt(operand.get(*it)));
  return result;
}

SmallVector<Tensor> scatterOp(
    ArrayRef<Tensor> inputs, const Tensor &scatterIndices,
    ArrayRef<Tensor> updates, const Axes &updateWindowDims,
    const Axes &insertedWindowDims, const Axes &inputBatchingDims,
    const Axes &scatterIndicesBatchingDims,
    const Axes &scatterDimsToOperandDims, Axis indexVectorDim,
    Region &updateComputation, Process *process, Scope &scope,
    ArrayRef<ShapedType> resultTypes) {
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
    auto startIndex = evalIndex(sliceOp(scatterIndices, startIndicesIndex));

    Index fullStartIndex(inputs[0].getRank(), 0);
    for (auto dInput : inputs[0].getAxes()) {
      auto dStartIt = llvm::find(scatterDimsToOperandDims, dInput);
      if (dStartIt == scatterDimsToOperandDims.end()) continue;
      auto dStart = dStartIt - scatterDimsToOperandDims.begin();
      fullStartIndex[dInput] = startIndex[dStart];
    }

    Index fullBatchingIndex(inputs[0].getRank(), 0);
    for (auto dInput : inputs[0].getAxes()) {
      auto dBatchingIt = llvm::find(inputBatchingDims, dInput);
      if (dBatchingIt == inputBatchingDims.end()) continue;
      auto iBatching = dBatchingIt - inputBatchingDims.begin();
      auto dStart = scatterIndicesBatchingDims[iBatching];
      fullBatchingIndex[dInput] = startIndicesIndex[dStart];
    }

    Index updateWindowIndex;
    for (auto d : updateWindowDims) updateWindowIndex.push_back(updateIndex[d]);

    Index fullWindowIndex(inputs[0].getRank(), 0);
    for (size_t i = 0, wi = 0; i < fullWindowIndex.size(); ++i) {
      if (llvm::is_contained(insertedWindowDims, i) ||
          llvm::is_contained(inputBatchingDims, i))
        continue;
      fullWindowIndex[i] = updateWindowIndex[wi++];
    }

    auto resultIndex = fullStartIndex + fullBatchingIndex + fullWindowIndex;
    if (!resultIndex.inBounds(results[0].getShape())) continue;

    SmallVector<InterpreterValue> updateComputationArgs;
    for (const auto &result : results)
      updateComputationArgs.push_back(constant(result.get(resultIndex)));
    for (const auto &update : updates)
      updateComputationArgs.push_back(constant(update.get(updateIndex)));

    auto updatedValues = eval(updateComputation, updateComputationArgs,
                              /*fallback=*/nullptr, process, &scope);
    for (auto [result, updatedValue] : llvm::zip(results, updatedValues))
      result.set(resultIndex, updatedValue.getTensor().get({}));
  }

  return results;
}

Tensor selectAndScatterOp(const Tensor &operand, const Tensor &source,
                          const Tensor &initValue,
                          const Sizes &windowDimensions,
                          const Sizes &windowStrides, const Sizes &paddingLow,
                          Region &select, Region &scatter, Process *process,
                          Scope &scope, ShapedType resultType) {
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

      InterpreterValue selectedInterpreterVal(constant(selectedVal.value()));
      InterpreterValue currInterpreterVal(constant(currVal));
      auto selectResult =
          eval(select, {selectedInterpreterVal, currInterpreterVal},
               /*fallback=*/nullptr, process, &scope);

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
        auto reducedResult =
            reduceOp({sourceValues}, {initValue}, {0}, scatter, process, scope);
        result.set(operandIndex, reducedResult[0].get({}));
      }
    });
  }
  return result;
}

Tensor selectOp(const Tensor &pred, const Tensor &onTrue, const Tensor &onFalse,
                ShapedType resultType) {
  Tensor result(resultType);
  for (auto it = result.index_begin(); it != result.index_end(); ++it) {
    Element predValue = pred.getRank() != 0 ? pred.get(*it) : pred.get({});
    result.set(
        *it, predValue.getBooleanValue() ? onTrue.get(*it) : onFalse.get(*it));
  }
  return result;
}

Token sendOp(ArrayRef<Tensor> inputs, Token token, ChannelId channelId,
             Process *process) {
  process->send(inputs, channelId);
  return token;
}

Tensor shiftLeftOp(const Tensor &lhs, const Tensor &rhs,
                   ShapedType resultType) {
  Tensor result(resultType);
  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it, shiftLeft(lhs.get(*it), rhs.get(*it)));
  return result;
}

Tensor shiftRightArithmeticOp(const Tensor &lhs, const Tensor &rhs,
                              ShapedType resultType) {
  Tensor result(resultType);
  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it, shiftRightArithmetic(lhs.get(*it), rhs.get(*it)));
  return result;
}

Tensor shiftRightLogicalOp(const Tensor &lhs, const Tensor &rhs,
                           ShapedType resultType) {
  Tensor result(resultType);
  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it, shiftRightLogical(lhs.get(*it), rhs.get(*it)));
  return result;
}

Tensor signOp(const Tensor &operand, ShapedType resultType) {
  Tensor result(resultType);
  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it, sign(operand.get(*it)));
  return result;
}

Tensor sineOp(const Tensor &operand, ShapedType resultType) {
  Tensor result(resultType);
  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it, sine(operand.get(*it)));
  return result;
}

Tensor sliceOp(const Tensor &operand, const Sizes &startIndices,
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

SmallVector<Tensor> sortOp(ArrayRef<Tensor> inputs, Axis dimension,
                           bool isStable, Region &comparator, Process *process,
                           Scope &scope) {
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
        args.emplace_back(constant(input.get(lhsIndex)));
        args.emplace_back(constant(input.get(rhsIndex)));
      }
      auto comparatorResult =
          eval(comparator, args, /*fallback=*/nullptr, process, &scope);
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
    for (auto [resultHandle, inputHandle] : llvm::enumerate(inputsTogether)) {
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

Tensor sqrtOp(const Tensor &operand, ShapedType resultType) {
  Tensor result(resultType);
  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it, sqrt(operand.get(*it)));
  return result;
}

Tensor subtractOp(const Tensor &lhs, const Tensor &rhs, ShapedType resultType) {
  Tensor result(resultType);
  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it, lhs.get(*it) - rhs.get(*it));
  return result;
}

Tensor tanOp(const Tensor &operand, ShapedType resultType) {
  Tensor result(resultType);
  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it, tan(operand.get(*it)));
  return result;
}

Tensor tanhOp(const Tensor &operand, ShapedType resultType) {
  Tensor result(resultType);
  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it, tanh(operand.get(*it)));
  return result;
}

Tensor transposeOp(const Tensor &operand, const Axes &permutation,
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

Tuple tupleOp(ArrayRef<InterpreterValue> val, TupleType resultType) {
  return Tuple(val, resultType);
}

SmallVector<InterpreterValue> whileOp(SmallVector<InterpreterValue> operand,
                                      Region &cond, Region &body,
                                      InterpreterFallback *fallback,
                                      Process *process, Scope &scope) {
  SmallVector<InterpreterValue> results(operand);

  auto condResults = eval(cond, operand, fallback, process, &scope);

  while (condResults[0].getTensor().get({}).getBooleanValue()) {
    results = eval(body, results, fallback, process, &scope);
    condResults = eval(cond, results, fallback, process, &scope);
  }

  return results;
}

Tensor xorOp(const Tensor &lhs, const Tensor &rhs, ShapedType resultType) {
  Tensor result(resultType);
  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it, lhs.get(*it) ^ rhs.get(*it));
  return result;
}

}  // namespace stablehlo
}  // namespace mlir
