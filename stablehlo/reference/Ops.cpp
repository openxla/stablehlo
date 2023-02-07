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
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/DebugStringHelper.h"
#include "stablehlo/reference/Element.h"
#include "stablehlo/reference/Errors.h"
#include "stablehlo/reference/Types.h"

namespace mlir {
namespace stablehlo {

namespace {

// Appies the permutation `perm` to an array `array` where perm[i] indicates the
// location where the current array[i] goes.
SmallVector<int64_t> permute(ArrayRef<int64_t> array, ArrayRef<int64_t> perm) {
  SmallVector<int64_t> result(array.size());
  for (size_t i = 0; i < array.size(); i++) result[i] = array[perm[i]];
  return result;
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

Tensor evalCeilOp(const Tensor &operand, Type resultType) {
  Tensor result(resultType);
  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it, ceil(operand.get(*it)));
  return result;
}

Tensor evalConstantOp(ElementsAttr value) {
  return makeTensor(value.cast<DenseElementsAttr>());
}

Tensor evalCosineOp(const Tensor &operand, Type resultType) {
  Tensor result(resultType);
  for (auto it = result.index_begin(); it != result.index_end(); ++it)
    result.set(*it, cosine(operand.get(*it)));
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

Tensor evalXorOp(const Tensor &lhs, const Tensor &rhs, Type resultType) {
  Tensor result(resultType);
  for (auto it = lhs.index_begin(); it != lhs.index_end(); ++it)
    result.set(*it, lhs.get(*it) ^ rhs.get(*it));
  return result;
}

}  // namespace stablehlo
}  // namespace mlir
