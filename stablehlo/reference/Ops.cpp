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

// Applies the permutation `perm` to an array `array` where perm[i] indicates
// the location where the current array[i] goes.
SmallVector<int64_t> permute(ArrayRef<int64_t> array, ArrayRef<int64_t> perm) {
  SmallVector<int64_t> result(array.size());
  for (size_t i = 0; i < array.size(); i++) result[i] = array[perm[i]];
  return result;
}

SmallVector<int64_t> addIndices(ArrayRef<int64_t> lhs, ArrayRef<int64_t> rhs) {
  SmallVector<int64_t> combined;
  for (auto [lhsIdx, rhsIdx] : llvm::zip(lhs, rhs))
    combined.push_back(lhsIdx + rhsIdx);
  return combined;
}

}  // namespace

Tensor evalAddOp(const Tensor &lhs, const Tensor &rhs, Type resultType) {
  Tensor result(resultType);
  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it, lhs.get(*it) + rhs.get(*it));
  return result;
}

Tensor evalAndOp(const Tensor &lhs, const Tensor &rhs, Type resultType) {
  Tensor result(resultType);
  for (auto it = lhs.index_begin(); it != lhs.index_end(); ++it)
    result.set(*it, lhs.get(*it) & rhs.get(*it));
  return result;
}

Tensor evalBroadcastInDimOp(const Tensor &operand,
                            ArrayRef<int64_t> broadcastDimensions,
                            Type resultType) {
  Tensor result(resultType);
  auto operandShape = operand.getType().getShape();
  for (auto resultIt = result.index_begin(); resultIt != result.index_end();
       ++resultIt) {
    SmallVector<int64_t> operandIdx;
    for (auto [operandDim, resultDim] : llvm::enumerate(broadcastDimensions))
      operandIdx.push_back(
          operandShape[operandDim] == 1 ? 0 : (*resultIt)[resultDim]);
    result.set(*resultIt, operand.get(operandIdx));
  }
  return result;
}

Tensor evalCeilOp(const Tensor &operand, Type resultType) {
  Tensor result(resultType);
  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it, ceil(operand.get(*it)));
  return result;
}

Tensor evalConstantOp(ElementsAttr value) {
  return makeTensor(value.cast<DenseElementsAttr>());
}

// This is an simplified implementation of convert op semantics dealing only
// with integer to bool conversion. To be updated as part of #969.
Tensor evalConvertOp(const Tensor &operand, Type resultType) {
  Tensor result(resultType);
  Type elType = result.getType().getElementType();
  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it, Element(elType,
                            operand.get(*it).getIntegerValue().getBoolValue()));
  return result;
}

Tensor evalCosineOp(const Tensor &operand, Type resultType) {
  Tensor result(resultType);
  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it, cosine(operand.get(*it)));
  return result;
}

Tensor evalDynamicUpdateSliceOp(const Tensor &operand, const Tensor &update,
                                ArrayRef<Tensor> startIndices,
                                Type resultType) {
  Tensor result(resultType);
  auto operandShape = operand.getType().getShape();
  auto updateShape = update.getType().getShape();
  SmallVector<int64_t> adjustedStartIndices;
  for (size_t i = 0; i < startIndices.size(); ++i)
    adjustedStartIndices.push_back(std::min(
        std::max(startIndices[i].get({}).getIntegerValue().getSExtValue(), 0l),
        operandShape[i] - updateShape[i]));
  for (auto resultIt = result.index_begin(); resultIt != result.index_end();
       ++resultIt)
    result.set(*resultIt, operand.get(*resultIt));
  for (auto updateIt = update.index_begin(); updateIt != update.index_end();
       ++updateIt)
    result.set(addIndices(*updateIt, adjustedStartIndices),
               update.get(*updateIt));
  return result;
}

Tensor evalFloorOp(const Tensor &operand, Type resultType) {
  Tensor result(resultType);
  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it, floor(operand.get(*it)));
  return result;
}

Tensor evalIotaOp(int64_t iotaDimension, Type resultType) {
  Tensor result(resultType);
  Type elType = result.getType().getElementType();
  for (auto it = result.index_begin(); it != result.index_end(); ++it) {
    auto iota = (*it)[iotaDimension];
    if (isSupportedSignedIntegerType(elType)) {
      result.set(*it, Element(elType, APInt(elType.getIntOrFloatBitWidth(),
                                            iota, /*isSigned=*/true)));
    } else if (isSupportedUnsignedIntegerType(elType)) {
      result.set(*it, Element(elType, APInt(elType.getIntOrFloatBitWidth(),
                                            iota, /*isSigned=*/false)));
    } else if (isSupportedFloatType(elType)) {
      APFloat val = APFloat((double)iota);
      bool roundingErr;
      val.convert(elType.cast<FloatType>().getFloatSemantics(),
                  APFloat::rmNearestTiesToEven, &roundingErr);
      result.set(*it, Element(elType, val));
    } else if (isSupportedComplexType(elType)) {
      APFloat real((double)iota);
      APFloat imag((double)0.0);
      FloatType flType =
          elType.cast<ComplexType>().getElementType().cast<FloatType>();
      bool roundingErr;
      real.convert(flType.getFloatSemantics(), APFloat::rmNearestTiesToEven,
                   &roundingErr);
      imag.convert(flType.getFloatSemantics(), APFloat::rmNearestTiesToEven,
                   &roundingErr);
      result.set(*it, Element(elType, std::complex<APFloat>(real, imag)));
    } else {
      report_fatal_error(invalidArgument("Unsupported element type: %s",
                                         debugString(elType).c_str()));
    }
  }
  return result;
}

Tensor evalMaxOp(const Tensor &lhs, const Tensor &rhs, Type resultType) {
  Tensor result(resultType);
  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it, max(lhs.get(*it), rhs.get(*it)));
  return result;
}

Tensor evalMinOp(const Tensor &lhs, const Tensor &rhs, Type resultType) {
  Tensor result(resultType);
  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it, min(lhs.get(*it), rhs.get(*it)));
  return result;
}

Tensor evalMultiplyOp(const Tensor &lhs, const Tensor &rhs, Type resultType) {
  Tensor result(resultType);
  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it, lhs.get(*it) * rhs.get(*it));
  return result;
}

Tensor evalNegOp(const Tensor &operand, Type resultType) {
  Tensor result(resultType);
  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it, -operand.get(*it));
  return result;
}

Tensor evalNotOp(const Tensor &operand, Type resultType) {
  Tensor result(resultType);
  for (auto it = operand.index_begin(); it != operand.index_end(); ++it)
    result.set(*it, ~operand.get(*it));
  return result;
}

Tensor evalOrOp(const Tensor &lhs, const Tensor &rhs, Type resultType) {
  Tensor result(resultType);
  for (auto it = lhs.index_begin(); it != lhs.index_end(); ++it)
    result.set(*it, lhs.get(*it) | rhs.get(*it));
  return result;
}

Tensor evalPadOp(const Tensor &operand, const Tensor &paddingValue,
                 ArrayRef<int64_t> edgePaddingLow,
                 ArrayRef<int64_t> interiorPadding, Type resultType) {
  Tensor result(resultType);
  for (auto resultIt = result.index_begin(); resultIt != result.index_end();
       ++resultIt)
    result.set(*resultIt, paddingValue.get({}));
  for (auto operandIt = operand.index_begin(); operandIt != operand.index_end();
       ++operandIt) {
    SmallVector<int64_t> resultIdx(result.getType().getRank());
    for (auto i = 0; i < operand.getType().getRank(); ++i)
      resultIdx[i] =
          edgePaddingLow[i] + (*operandIt)[i] * (interiorPadding[i] + 1);
    if (succeeded(verifyIndex(result.getType().getShape(), resultIdx)))
      result.set(resultIdx, operand.get(*operandIt));
  }
  return result;
}

Tensor evalReshapeOp(const Tensor &operand, Type resultType) {
  Tensor result(resultType);
  for (auto resultIt = result.index_begin(), operandIt = operand.index_begin();
       resultIt != result.index_end(); ++resultIt, ++operandIt)
    result.set(*resultIt, operand.get(*operandIt));
  return result;
}

Tensor evalReverseOp(const Tensor &operand, ArrayRef<int64_t> dimensions,
                     Type resultType) {
  Tensor result(resultType);
  auto resultShape = result.getType().getShape();
  for (auto resultIt = result.index_begin(); resultIt != result.index_end();
       ++resultIt) {
    SmallVector<int64_t> operandIdx(*resultIt);
    for (auto dim : dimensions)
      operandIdx[dim] = (resultShape[dim] - 1) - operandIdx[dim];
    result.set(*resultIt, operand.get(operandIdx));
  }
  return result;
}

Tensor evalSineOp(const Tensor &operand, Type resultType) {
  Tensor result(resultType);
  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it, sine(operand.get(*it)));
  return result;
}

Tensor evalSliceOp(const Tensor &operand, ArrayRef<int64_t> startIndices,
                   ArrayRef<int64_t> strides, Type resultType) {
  Tensor result(resultType);
  for (auto resultIt = result.index_begin(); resultIt != result.index_end();
       ++resultIt) {
    SmallVector<int64_t> operandIdx;
    for (auto dim = 0; dim < operand.getType().getRank(); ++dim)
      operandIdx.push_back(startIndices[dim] + (*resultIt)[dim] * strides[dim]);
    result.set(*resultIt, operand.get(operandIdx));
  }
  return result;
}

Tensor evalSubtractOp(const Tensor &lhs, const Tensor &rhs, Type resultType) {
  Tensor result(resultType);
  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it, lhs.get(*it) - rhs.get(*it));
  return result;
}

Tensor evalTanhOp(const Tensor &operand, Type resultType) {
  Tensor result(resultType);
  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it, tanh(operand.get(*it)));
  return result;
}

Tensor evalTransposeOp(const Tensor &operand, ArrayRef<int64_t> permutation,
                       Type resultType) {
  Tensor result(resultType);
  for (auto operandIt = operand.index_begin(); operandIt != operand.index_end();
       ++operandIt) {
    auto resultIndex = permute(*operandIt, permutation);
    result.set(resultIndex, operand.get(*operandIt));
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

Tensor evalXorOp(const Tensor &lhs, const Tensor &rhs, Type resultType) {
  Tensor result(resultType);
  for (auto it = lhs.index_begin(); it != lhs.index_end(); ++it)
    result.set(*it, lhs.get(*it) ^ rhs.get(*it));
  return result;
}

SmallVector<Tensor> eval(Region &region, ArrayRef<Tensor> args, Scope *parent) {
  Block &block = region.front();
  if (block.getArguments().size() != args.size())
    report_fatal_error(invalidArgument(
        "Expected same amount of block arguments and runtime arguments (%d)",
        args.size()));

  Scope scope(parent);
  scope.add(block.getArguments(), args);

  for (Operation &op : block) {
    if (auto addOp = dyn_cast<AddOp>(op)) {
      Tensor runtimeLhs = scope.find(addOp.getLhs());
      Tensor runtimeRhs = scope.find(addOp.getRhs());
      Tensor runtimeResult = evalAddOp(runtimeLhs, runtimeRhs, addOp.getType());
      scope.add(op.getResults(), {runtimeResult});
    } else if (auto andOp = dyn_cast<AndOp>(op)) {
      Tensor runtimeLhs = scope.find(andOp.getLhs());
      Tensor runtimeRhs = scope.find(andOp.getRhs());
      Tensor runtimeResult = evalAndOp(runtimeLhs, runtimeRhs, andOp.getType());
      scope.add(op.getResults(), {runtimeResult});
    } else if (auto broadcastInDimOp = dyn_cast<BroadcastInDimOp>(op)) {
      Tensor runtimeOperand = scope.find(broadcastInDimOp.getOperand());
      auto broadcastDimensions = llvm::to_vector(
          broadcastInDimOp.getBroadcastDimensions().getValues<int64_t>());
      Tensor runtimeResult = evalBroadcastInDimOp(
          runtimeOperand, broadcastDimensions, broadcastInDimOp.getType());
      scope.add(op.getResults(), {runtimeResult});
    } else if (auto ceilOp = dyn_cast<CeilOp>(op)) {
      Tensor runtimeOperand = scope.find(ceilOp.getOperand());
      Tensor runtimeResult = evalCeilOp(runtimeOperand, ceilOp.getType());
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
    } else if (auto dynamicUpdateSliceOp = dyn_cast<DynamicUpdateSliceOp>(op)) {
      Tensor runtimeOperand = scope.find(dynamicUpdateSliceOp.getOperand());
      Tensor runtimeUpdate = scope.find(dynamicUpdateSliceOp.getUpdate());
      SmallVector<Tensor> runtimeStartIndices =
          scope.find(dynamicUpdateSliceOp.getStartIndices());
      Tensor runtimeResult = evalDynamicUpdateSliceOp(
          runtimeOperand, runtimeUpdate, runtimeStartIndices,
          dynamicUpdateSliceOp.getType());
      scope.add(op.getResults(), {runtimeResult});
    } else if (auto floorOp = dyn_cast<FloorOp>(op)) {
      Tensor runtimeOperand = scope.find(floorOp.getOperand());
      Tensor runtimeResult = evalFloorOp(runtimeOperand, floorOp.getType());
      scope.add(op.getResults(), {runtimeResult});
    } else if (auto iotaOp = dyn_cast<IotaOp>(op)) {
      Tensor runtimeResult =
          evalIotaOp(iotaOp.getIotaDimension(), iotaOp.getType());
      scope.add(op.getResults(), {runtimeResult});
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
      auto edgePaddingLow =
          llvm::to_vector(padOp.getEdgePaddingLow().getValues<int64_t>());
      auto interiorPadding =
          llvm::to_vector(padOp.getInteriorPadding().getValues<int64_t>());
      Tensor runtimeResult =
          evalPadOp(runtimeOperand, runtimePaddingValue, edgePaddingLow,
                    interiorPadding, padOp.getType());
      scope.add(op.getResults(), {runtimeResult});
    } else if (auto whileOp = dyn_cast<WhileOp>(op)) {
      SmallVector<Value> runtimeOperands(whileOp.getOperand().begin(),
                                         whileOp.getOperand().end());
      auto runtimeInputs = scope.find(runtimeOperands);
      auto runtimeResults = evalWhileOp(runtimeInputs, whileOp.getCond(),
                                        whileOp.getBody(), scope);
      scope.add(op.getResults(), runtimeResults);
    } else if (auto reshapeOp = dyn_cast<ReshapeOp>(op)) {
      Tensor runtimeOperand = scope.find(reshapeOp.getOperand());
      Tensor runtimeResult = evalReshapeOp(runtimeOperand, reshapeOp.getType());
      scope.add(op.getResults(), {runtimeResult});
    } else if (auto reverseOp = dyn_cast<ReverseOp>(op)) {
      Tensor runtimeOperand = scope.find(reverseOp.getOperand());
      auto dimensions =
          llvm::to_vector(reverseOp.getDimensions().getValues<int64_t>());
      Tensor runtimeResult =
          evalReverseOp(runtimeOperand, dimensions, reverseOp.getType());
      scope.add(op.getResults(), {runtimeResult});
    } else if (auto returnOp = dyn_cast<func::ReturnOp>(op)) {
      return scope.find(returnOp.getOperands());
    } else if (auto returnOp = dyn_cast<ReturnOp>(op)) {
      return scope.find(returnOp.getResults());
    } else if (auto sineOp = dyn_cast<SineOp>(op)) {
      Tensor runtimeOperand = scope.find(sineOp.getOperand());
      Tensor runtimeResult = evalSineOp(runtimeOperand, sineOp.getType());
      scope.add(op.getResults(), {runtimeResult});
    } else if (auto sliceOp = dyn_cast<SliceOp>(op)) {
      Tensor runtimeOperand = scope.find(sliceOp.getOperand());
      auto startIndices =
          llvm::to_vector(sliceOp.getStartIndices().getValues<int64_t>());
      auto strides = llvm::to_vector(sliceOp.getStrides().getValues<int64_t>());
      Tensor runtimeResult =
          evalSliceOp(runtimeOperand, startIndices, strides, sliceOp.getType());
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
      auto permutation =
          llvm::to_vector(transposeOp.getPermutation().getValues<int64_t>());
      Tensor runtimeResult =
          evalTransposeOp(runtimeOperand, permutation, transposeOp.getType());
      scope.add(op.getResults(), {runtimeResult});
    } else if (auto xorOp = dyn_cast<XorOp>(op)) {
      Tensor runtimeLhs = scope.find(xorOp.getLhs());
      Tensor runtimeRhs = scope.find(xorOp.getRhs());
      Tensor runtimeResult = evalXorOp(runtimeLhs, runtimeRhs, xorOp.getType());
      scope.add(op.getResults(), {runtimeResult});
    } else {
      report_fatal_error(
          invalidArgument("Unsupported op: %s", debugString(op).c_str()));
    }
  }

  llvm::report_fatal_error("Expected a terminator when evaluating a region");
}

}  // namespace stablehlo
}  // namespace mlir
