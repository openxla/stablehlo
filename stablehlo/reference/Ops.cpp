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
#include "stablehlo/reference/Types.h"

namespace mlir {
namespace stablehlo {
namespace {
template <typename... Ts>
inline llvm::Error invalidArgument(char const *Fmt, const Ts &...Vals) {
  return createStringError(llvm::errc::invalid_argument, Fmt, Vals...);
}
}  // namespace

namespace {

// Appies the permutation `perm` to an array `array` where perm[i] indicates the
// location where the current array[i] goes.
std::vector<int64_t> permute(ArrayRef<int64_t> array, ArrayRef<int64_t> perm) {
  std::vector<int64_t> result(array.size());
  for (size_t i = 0; i < array.size(); i++) result[i] = array[perm[i]];
  return result;
}

}  // namespace

Tensor eval(AddOp op, const Tensor &lhs, const Tensor &rhs) {
  Tensor result(op.getType());
  for (auto it = result.index_begin(); it != result.index_end(); ++it) {
    result.set(*it, lhs.get(*it) + rhs.get(*it));
  }
  return result;
}

Tensor eval(AndOp op, const Tensor &lhs, const Tensor &rhs) {
  Tensor result(op.getType());
  for (auto it = lhs.index_begin(); it != lhs.index_end(); ++it) {
    result.set(*it, lhs.get(*it) & rhs.get(*it));
  }
  return result;
}

Tensor eval(CeilOp op, const Tensor &operand) {
  Tensor result(op.getType());
  for (auto it = result.index_begin(); it != result.index_end(); ++it) {
    result.set(*it, ceil(operand.get(*it)));
  }
  return result;
}

Tensor eval(ConstantOp op) {
  return makeTensor(op.getValue().cast<DenseElementsAttr>());
}

Tensor eval(CosineOp op, const Tensor &operand) {
  Tensor result(op.getType());
  for (auto it = result.index_begin(); it != result.index_end(); ++it) {
    result.set(*it, cosine(operand.get(*it)));
  }
  return result;
}

Tensor eval(DotGeneralOp op, const Tensor &lhs, const Tensor &rhs) {
  Tensor result(op.getType());
  ArrayRef<int64_t> lhsShape = lhs.getType().getShape();
  ArrayRef<int64_t> lhsBatchingDims =
      op.getDotDimensionNumbers().getLhsBatchingDimensions();
  ArrayRef<int64_t> rhsBatchingDims =
      op.getDotDimensionNumbers().getRhsBatchingDimensions();
  ArrayRef<int64_t> lhsContractingDims =
      op.getDotDimensionNumbers().getLhsContractingDimensions();
  ArrayRef<int64_t> rhsContractingDims =
      op.getDotDimensionNumbers().getRhsContractingDimensions();
  std::vector<int64_t> lhsNonContractingDims;
  for (int64_t i = 0; i < lhs.getType().getRank(); ++i)
    if (std::find(lhsBatchingDims.begin(), lhsBatchingDims.end(), i) ==
            lhsBatchingDims.end() &&
        std::find(lhsContractingDims.begin(), lhsContractingDims.end(), i) ==
            lhsContractingDims.end())
      lhsNonContractingDims.push_back(i);
  std::vector<int64_t> rhsNonContractingDims;
  for (int64_t i = 0; i < rhs.getType().getRank(); ++i)
    if (std::find(rhsBatchingDims.begin(), rhsBatchingDims.end(), i) ==
            rhsBatchingDims.end() &&
        std::find(rhsContractingDims.begin(), rhsContractingDims.end(), i) ==
            rhsContractingDims.end())
      rhsNonContractingDims.push_back(i);
  std::vector<int64_t> contractingDimSizes;
  for (uint64_t i = 0; i < lhsContractingDims.size(); ++i)
    contractingDimSizes.push_back(lhsShape[lhsContractingDims[i]]);
  auto totalContractingSize = 1;
  for (uint64_t i = 0; i < contractingDimSizes.size(); ++i)
    totalContractingSize *= contractingDimSizes[i];
  for (auto resIt = result.index_begin(); resIt != result.index_end();
       ++resIt) {
    Type resElTy =
        op->getResultTypes().front().dyn_cast<ShapedType>().getElementType();
    Element sum;
    if (isSupportedSignedIntegerType(resElTy)) {
      sum = Element(resElTy, APInt(resElTy.getIntOrFloatBitWidth(), 0,
                                   /*isSigned=*/true));
    } else if (isSupportedUnsignedIntegerType(resElTy)) {
      sum = Element(resElTy, APInt(resElTy.getIntOrFloatBitWidth(), 0,
                                   /*isSigned=*/false));
    } else if (isSupportedBooleanType(resElTy)) {
      sum = Element(resElTy, false);
    } else if (isSupportedFloatType(resElTy)) {
      APFloat val((double)0.0);
      bool roundingErr;
      val.convert(resElTy.cast<FloatType>().getFloatSemantics(),
                  APFloat::rmNearestTiesToEven, &roundingErr);
      sum = Element(resElTy, val);
    } else if (isSupportedComplexType(resElTy)) {
      APFloat real((double)0.0);
      APFloat imag((double)0.0);
      FloatType flType =
          resElTy.cast<ComplexType>().getElementType().cast<FloatType>();
      bool roundingErr;
      real.convert(flType.getFloatSemantics(), APFloat::rmNearestTiesToEven,
                   &roundingErr);
      imag.convert(flType.getFloatSemantics(), APFloat::rmNearestTiesToEven,
                   &roundingErr);
      sum = Element(resElTy, std::complex<APFloat>(real, imag));
    }
    std::vector<int64_t> lhsIdx(lhs.getType().getRank());
    std::vector<int64_t> rhsIdx(rhs.getType().getRank());
    std::vector<int64_t> resIdx = (*resIt).vec();
    int64_t idx = 0;
    for (uint64_t i = 0; i < lhsBatchingDims.size(); ++i, ++idx) {
      lhsIdx[lhsBatchingDims[i]] = resIdx[idx];
      rhsIdx[rhsBatchingDims[i]] = resIdx[idx];
    }
    for (uint64_t i = 0; i < lhsNonContractingDims.size(); i++)
      lhsIdx[lhsNonContractingDims[i]] = resIdx[idx++];
    for (uint64_t i = 0; i < rhsNonContractingDims.size(); i++)
      rhsIdx[rhsNonContractingDims[i]] = resIdx[idx++];
    for (int64_t k = 0; k < totalContractingSize; ++k) {
      sum = sum + lhs.get(lhsIdx) * rhs.get(rhsIdx);
      if (!contractingDimSizes.empty()) {
        for (int64_t i = contractingDimSizes.size() - 1; i >= 0; --i) {
          lhsIdx[lhsContractingDims[i]]++;
          rhsIdx[rhsContractingDims[i]]++;
          if (lhsIdx[lhsContractingDims[i]] != contractingDimSizes[i]) break;
          lhsIdx[lhsContractingDims[i]] = 0;
          rhsIdx[rhsContractingDims[i]] = 0;
        }
      }
    }
    result.set(resIdx, sum);
  }
  return result;
}

Tensor eval(FloorOp op, const Tensor &operand) {
  Tensor result(op.getType());
  for (auto it = result.index_begin(); it != result.index_end(); ++it) {
    result.set(*it, floor(operand.get(*it)));
  }
  return result;
}

Tensor eval(IotaOp op) {
  Tensor result(op.getType());
  Type elType = result.getType().getElementType();
  uint64_t iotaDimension = op.getIotaDimension();
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

Tensor eval(MaxOp op, const Tensor &lhs, const Tensor &rhs) {
  Tensor result(op.getType());
  for (auto it = result.index_begin(); it != result.index_end(); ++it) {
    result.set(*it, max(lhs.get(*it), rhs.get(*it)));
  }
  return result;
}

Tensor eval(MinOp op, const Tensor &lhs, const Tensor &rhs) {
  Tensor result(op.getType());
  for (auto it = result.index_begin(); it != result.index_end(); ++it) {
    result.set(*it, min(lhs.get(*it), rhs.get(*it)));
  }
  return result;
}

Tensor eval(MulOp op, const Tensor &lhs, const Tensor &rhs) {
  Tensor result(op.getType());
  for (auto it = result.index_begin(); it != result.index_end(); ++it) {
    result.set(*it, lhs.get(*it) * rhs.get(*it));
  }
  return result;
}

Tensor eval(NegOp op, const Tensor &operand) {
  Tensor result(op.getType());
  for (auto it = result.index_begin(); it != result.index_end(); ++it) {
    result.set(*it, -operand.get(*it));
  }
  return result;
}

Tensor eval(NotOp op, const Tensor &operand) {
  Tensor result(op.getType());
  for (auto it = operand.index_begin(); it != operand.index_end(); ++it) {
    result.set(*it, ~operand.get(*it));
  }
  return result;
}

Tensor eval(OrOp op, const Tensor &lhs, const Tensor &rhs) {
  Tensor result(op.getType());
  for (auto it = lhs.index_begin(); it != lhs.index_end(); ++it) {
    result.set(*it, lhs.get(*it) | rhs.get(*it));
  }
  return result;
}

Tensor eval(ReshapeOp op, const Tensor &operand) {
  Tensor result(op.getType());
  for (auto resultIt = result.index_begin(), operandIt = operand.index_begin();
       resultIt != result.index_end(); ++resultIt, ++operandIt) {
    result.set(*resultIt, operand.get(*operandIt));
  }
  return result;
}

Tensor eval(SineOp op, const Tensor &operand) {
  Tensor result(op.getType());
  for (auto it = result.index_begin(); it != result.index_end(); ++it) {
    result.set(*it, sine(operand.get(*it)));
  }
  return result;
}

Tensor eval(SubtractOp op, const Tensor &lhs, const Tensor &rhs) {
  Tensor result(op.getType());
  for (auto it = result.index_begin(); it != result.index_end(); ++it) {
    result.set(*it, lhs.get(*it) - rhs.get(*it));
  }
  return result;
}

Tensor eval(TanhOp op, const Tensor &operand) {
  Tensor result(op.getType());
  for (auto it = result.index_begin(); it != result.index_end(); ++it) {
    result.set(*it, tanh(operand.get(*it)));
  }
  return result;
}

Tensor eval(TransposeOp op, const Tensor &operand) {
  Tensor result(op.getType());
  for (auto operandIt = operand.index_begin(); operandIt != operand.index_end();
       ++operandIt) {
    auto resultIndex = permute(
        *operandIt, llvm::to_vector(op.getPermutation().getValues<int64_t>()));
    result.set(resultIndex, operand.get(*operandIt));
  }
  return result;
}

Tensor eval(XorOp op, const Tensor &lhs, const Tensor &rhs) {
  Tensor result(op.getType());
  for (auto it = lhs.index_begin(); it != lhs.index_end(); ++it) {
    result.set(*it, lhs.get(*it) ^ rhs.get(*it));
  }
  return result;
}

}  // namespace stablehlo
}  // namespace mlir
