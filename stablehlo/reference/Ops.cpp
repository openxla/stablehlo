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
  for (auto it = lhs.index_begin(); it != lhs.index_end(); ++it) {
    result.set(*it, lhs.get(*it) + rhs.get(*it));
  }
  return result;
}

Tensor eval(CeilOp op, const Tensor &operand) {
  Tensor result(op.getType());
  for (auto it = operand.index_begin(); it != operand.index_end(); ++it) {
    result.set(*it, ceil(operand.get(*it)));
  }
  return result;
}

Tensor eval(ConstantOp op) {
  return makeTensor(op.getValue().cast<DenseElementsAttr>());
}

Tensor eval(CosineOp op, const Tensor &operand) {
  Tensor result(op.getType());
  for (auto it = operand.index_begin(); it != operand.index_end(); ++it) {
    result.set(*it, cosine(operand.get(*it)));
  }
  return result;
}

Tensor eval(FloorOp op, const Tensor &operand) {
  Tensor result(op.getType());
  for (auto it = operand.index_begin(); it != operand.index_end(); ++it) {
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

Tensor eval(NegOp op, const Tensor &operand) {
  Tensor result(op.getType());
  for (auto it = operand.index_begin(); it != operand.index_end(); ++it) {
    result.set(*it, -operand.get(*it));
  }
  return result;
}

Tensor eval(ReshapeOp op, const Tensor &operand) {
  Tensor result(op.getType());
  for (auto operandIt = operand.index_begin(), resultIt = result.index_begin();
       operandIt != operand.index_end(); ++operandIt, ++resultIt) {
    result.set(*resultIt, operand.get(*operandIt));
  }
  return result;
}

Tensor eval(MaxOp op, const Tensor &lhs, const Tensor &rhs) {
  Tensor result(op.getType());
  for (auto it = lhs.index_begin(); it != lhs.index_end(); ++it) {
    result.set(*it, max(lhs.get(*it), rhs.get(*it)));
  }
  return result;
}

Tensor eval(MinOp op, const Tensor &lhs, const Tensor &rhs) {
  Tensor result(op.getType());
  for (auto it = lhs.index_begin(); it != lhs.index_end(); ++it) {
    result.set(*it, min(lhs.get(*it), rhs.get(*it)));
  }
  return result;
}

Tensor eval(SineOp op, const Tensor &operand) {
  Tensor result(op.getType());
  for (auto it = operand.index_begin(); it != operand.index_end(); ++it) {
    result.set(*it, sine(operand.get(*it)));
  }
  return result;
}

Tensor eval(SubtractOp op, const Tensor &lhs, const Tensor &rhs) {
  Tensor result(op.getType());
  for (auto it = lhs.index_begin(); it != lhs.index_end(); ++it) {
    result.set(*it, lhs.get(*it) - rhs.get(*it));
  }
  return result;
}

Tensor eval(TanhOp op, const Tensor &operand) {
  Tensor result(op.getType());
  for (auto it = operand.index_begin(); it != operand.index_end(); ++it) {
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

}  // namespace stablehlo
}  // namespace mlir
