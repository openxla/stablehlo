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

#include "Tensor.h"
#include "llvm/ADT/APFloat.h"
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
  ArrayRef<int64_t> shape = result.getType().getShape();
  Type type = result.getType().getElementType();
  if (!isSupportedIntegerType(type) && !isSupportedFloatType(type) &&
      !isSupportedComplexType(type))
    report_fatal_error(invalidArgument("Unsupported element type: %s",
                                       debugString(type).c_str()));
  uint64_t rightOfIota = 1;
  uint64_t iotaDimension = op.getIotaDimension();
  for (uint64_t i = 0; i < static_cast<uint64_t>(shape.size()); i++)
    rightOfIota *= i > iotaDimension ? shape[i] : 1;
  llvm::RoundingMode rm = APFloat::rmNearestTiesToEven;
  bool roundingErr;
  for (auto [i, iota, it] = std::tuple{0, 0, result.index_begin()};
       it != result.index_end(); ++it, ++i) {
    if (i != 0 && i % (rightOfIota) == 0) iota++;
    if (iota >= shape[iotaDimension]) iota = 0;
    if (isSupportedIntegerType(type)) {
      result.set(
          *it, Element(type, APInt(type.getIntOrFloatBitWidth(), iota, false)));
    } else if (isSupportedFloatType(type)) {
      APFloat val = APFloat((double)iota);
      if (type.isBF16()) val.convert(APFloat::BFloat(), rm, &roundingErr);
      if (type.isF16()) val.convert(APFloat::IEEEhalf(), rm, &roundingErr);
      if (type.isF32()) val.convert(APFloat::IEEEsingle(), rm, &roundingErr);
      if (type.isF64()) val.convert(APFloat::IEEEdouble(), rm, &roundingErr);
      result.set(*it, Element(type, val));
    } else if (isSupportedComplexType(type)) {
      APFloat real((double)iota);
      APFloat imag((double)0.0);
      Type elType = type.dyn_cast<ComplexType>().getElementType();
      if (elType.isF32()) {
        real.convert(APFloat::IEEEsingle(), rm, &roundingErr);
        imag.convert(APFloat::IEEEsingle(), rm, &roundingErr);
        result.set(*it, Element(type, std::complex<APFloat>(real, imag)));
      } else if (elType.isF64()) {
        result.set(*it, Element(type, std::complex<APFloat>(real, imag)));
      }
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
