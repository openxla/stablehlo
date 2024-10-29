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

#include "stablehlo/reference/Tensor.h"

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/reference/Element.h"
#include "stablehlo/reference/Errors.h"
#include "stablehlo/reference/Index.h"
#include "stablehlo/reference/Types.h"

namespace mlir {
namespace stablehlo {

namespace {

int64_t getSizeInBytes(Type type) {
  if (auto shapedType = dyn_cast<ShapedType>(type))
    return shapedType.getNumElements() *
           getSizeInBytes(shapedType.getElementType());

  if (type.isIntOrFloat())
    return std::max(type.getIntOrFloatBitWidth(), (unsigned)8) / 8;

  if (auto complexType = dyn_cast<mlir::ComplexType>(type))
    return getSizeInBytes(complexType.getElementType()) * 2;

  report_fatal_error(
      invalidArgument("Unsupported type: %s", debugString(type).c_str()));
}

// Flattens multi-dimensional index 'index' of a tensor to a linearized index
// into the underlying storage where elements are laid out in canonical order.
int64_t flattenIndex(const Sizes &shape, const Index &index) {
  if (!index.inBounds(shape))
    llvm::report_fatal_error(
        "Incompatible index and shape found while flattening index");

  int64_t idx = 0;
  if (shape.empty()) return idx;

  // Computes strides of a tensor shape: The number of locations in memory
  // between beginnings of successive array elements, measured in units of the
  // size of the array's elements.
  // Example: For a tensor shape [1,2,3], strides = [6,3,1]
  std::vector<int64_t> strides(shape.size());
  strides[shape.size() - 1] = 1;
  for (int i = shape.size() - 2; i >= 0; --i)
    strides[i] = strides[i + 1] * shape[i + 1];

  // Use the computed strides to flatten the multi-dimensional index 'index'
  // to a linearized index.
  // Example: For a tensor with shape [1,2,3], strides = [6,3,1], and index =
  // [0, 1, 2], the flattened index = 0*6 + 1*3 + 2*1 = 5
  for (size_t i = 0; i < index.size(); i++) idx += strides[i] * index[i];
  return idx;
}

}  // namespace

namespace detail {

Buffer::Buffer(ShapedType type)
    : type_(type),
      blob_(
          HeapAsmResourceBlob::allocate(getSizeInBytes(type), alignof(char))) {}

Buffer::Buffer(ShapedType type, AsmResourceBlob blob)
    : type_(type), blob_(std::move(blob)) {}

}  // namespace detail

Tensor::Tensor() {}

Tensor::Tensor(ShapedType type)
    : impl_(llvm::makeIntrusiveRefCnt<detail::Buffer>(type)) {}

Tensor::Tensor(ShapedType type, AsmResourceBlob blob)
    : impl_(llvm::makeIntrusiveRefCnt<detail::Buffer>(type, std::move(blob))) {}

Element Tensor::get(const Index &index) const {
  Type elementType = getType().getElementType();
  const char *elementPtr =
      impl_->getData().data() +
      getSizeInBytes(elementType) * flattenIndex(getShape(), index);

  // Handle floating-point types.
  if (isSupportedFloatType(elementType) &&
      cast<FloatType>(elementType).getWidth() <= 8) {
    auto elementData = reinterpret_cast<const uint8_t *>(elementPtr);
    auto floatTy = cast<FloatType>(elementType);
    return Element(elementType,
                   APFloat(floatTy.getFloatSemantics(),
                           APInt(floatTy.getWidth(), *elementData)));
  }
  if (elementType.isF16()) {
    auto elementData = reinterpret_cast<const uint16_t *>(elementPtr);
    return Element(elementType, APFloat(llvm::APFloatBase::IEEEhalf(),
                                        APInt(16, *elementData)));
  }

  if (elementType.isBF16()) {
    auto elementData = reinterpret_cast<const uint16_t *>(elementPtr);
    return Element(elementType, APFloat(llvm::APFloatBase::BFloat(),
                                        APInt(16, *elementData)));
  }

  if (elementType.isF32()) {
    auto elementData = reinterpret_cast<const float *>(elementPtr);
    return Element(elementType, APFloat(*elementData));
  }

  if (elementType.isF64()) {
    auto elementData = reinterpret_cast<const double *>(elementPtr);
    return Element(elementType, APFloat(*elementData));
  }

  // Handle integer types.
  // TODO(#22): StableHLO, as bootstrapped from MHLO, inherits signless
  // integers which was added in MHLO for legacy reasons. Going forward,
  // StableHLO will adopt signfull integer semantics with signed and unsigned
  // integer variants.
  if (isSupportedIntegerType(elementType)) {
    IntegerType intTy = cast<IntegerType>(elementType);

    if (elementType.isSignlessInteger(2) || elementType.isSignlessInteger(4) ||
        elementType.isSignlessInteger(8)) {
      auto elementData = reinterpret_cast<const int8_t *>(elementPtr);
      return Element(elementType, APInt(intTy.getWidth(), *elementData,
                                        intTy.isSignedInteger()));
    } else if (elementType.isSignlessInteger(16)) {
      auto elementData = reinterpret_cast<const int16_t *>(elementPtr);
      return Element(elementType, APInt(intTy.getWidth(), *elementData,
                                        intTy.isSignedInteger()));
    } else if (elementType.isSignlessInteger(32)) {
      auto elementData = reinterpret_cast<const int32_t *>(elementPtr);
      return Element(elementType, APInt(intTy.getWidth(), *elementData,
                                        intTy.isSignedInteger()));
    } else if (elementType.isSignlessInteger(64)) {
      auto elementData = reinterpret_cast<const int64_t *>(elementPtr);
      return Element(elementType, APInt(intTy.getWidth(), *elementData,
                                        intTy.isSignedInteger()));
    } else if (elementType.isUnsignedInteger(2) ||
               elementType.isUnsignedInteger(4) ||
               elementType.isUnsignedInteger(8)) {
      auto elementData = reinterpret_cast<const uint8_t *>(elementPtr);
      return Element(elementType, APInt(intTy.getWidth(), *elementData,
                                        intTy.isSignedInteger()));
    } else if (elementType.isUnsignedInteger(16)) {
      auto elementData = reinterpret_cast<const uint16_t *>(elementPtr);
      return Element(elementType, APInt(intTy.getWidth(), *elementData,
                                        intTy.isSignedInteger()));
    } else if (elementType.isUnsignedInteger(32)) {
      auto elementData = reinterpret_cast<const uint32_t *>(elementPtr);
      return Element(elementType, APInt(intTy.getWidth(), *elementData,
                                        intTy.isSignedInteger()));
    } else if (elementType.isUnsignedInteger(64)) {
      auto elementData = reinterpret_cast<const uint64_t *>(elementPtr);
      return Element(elementType, APInt(intTy.getWidth(), *elementData,
                                        intTy.isSignedInteger()));
    }
  }

  // Handle boolean type.
  if (isSupportedBooleanType(elementType)) {
    auto elementData = reinterpret_cast<const uint8_t *>(elementPtr);
    if (*elementData == 0) return Element(elementType, false);
    if (*elementData == 1) return Element(elementType, true);

    llvm::report_fatal_error("Unsupported boolean value");
  }

  // Handle complex types.
  if (isa<ComplexType>(elementType)) {
    auto complexElemTy = cast<ComplexType>(elementType).getElementType();

    if (complexElemTy.isF32()) {
      auto elementData =
          reinterpret_cast<const std::complex<float> *>(elementPtr);
      return Element(elementType,
                     std::complex<APFloat>(APFloat(elementData->real()),
                                           APFloat(elementData->imag())));
    }

    if (complexElemTy.isF64()) {
      auto elementData =
          reinterpret_cast<const std::complex<double> *>(elementPtr);
      return Element(elementType,
                     std::complex<APFloat>(APFloat(elementData->real()),
                                           APFloat(elementData->imag())));
    }
  }

  report_fatal_error(invalidArgument("Unsupported element type: %s",
                                     debugString(elementType).c_str()));
}

void Tensor::set(const Index &index, const Element &element) {
  Type elementType = getType().getElementType();
  char *elementPtr =
      impl_->getMutableData().data() +
      getSizeInBytes(elementType) * flattenIndex(getShape(), index);

  // Handle floating-point types.
  if (isSupportedFloatType(elementType) &&
      cast<FloatType>(elementType).getWidth() <= 8) {
    auto elementData = reinterpret_cast<uint8_t *>(elementPtr);
    auto value = element.getFloatValue();
    *elementData = (uint8_t)value.bitcastToAPInt().getZExtValue();
    return;
  }

  if (elementType.isF16() || elementType.isBF16()) {
    auto elementData = reinterpret_cast<uint16_t *>(elementPtr);
    auto value = element.getFloatValue();
    *elementData = (uint16_t)value.bitcastToAPInt().getZExtValue();
    return;
  }

  if (elementType.isF32()) {
    auto elementData = reinterpret_cast<float *>(elementPtr);
    auto value = element.getFloatValue();
    *elementData = value.convertToFloat();
    return;
  }

  if (elementType.isF64()) {
    auto elementData = reinterpret_cast<double *>(elementPtr);
    auto value = element.getFloatValue();
    *elementData = value.convertToDouble();
    return;
  }

  // Handle signed integer types.
  // TODO(#22): StableHLO, as bootstrapped from MHLO, inherits signless
  // integers which was added in MHLO for legacy reasons. Going forward,
  // StableHLO will adopt signfull integer semantics with signed and unsigned
  // integer variants.
  if (elementType.isSignlessInteger(2) || elementType.isSignlessInteger(4) ||
      elementType.isSignlessInteger(8)) {
    auto elementData = reinterpret_cast<int8_t *>(elementPtr);
    auto value = element.getIntegerValue();
    *elementData = (int8_t)value.getSExtValue();
    return;
  }

  if (elementType.isSignlessInteger(16)) {
    auto elementData = reinterpret_cast<int16_t *>(elementPtr);
    auto value = element.getIntegerValue();
    *elementData = (int16_t)value.getSExtValue();
    return;
  }

  if (elementType.isSignlessInteger(32)) {
    auto elementData = reinterpret_cast<int32_t *>(elementPtr);
    auto value = element.getIntegerValue();
    *elementData = (int32_t)value.getSExtValue();
    return;
  }

  if (elementType.isSignlessInteger(64)) {
    auto elementData = reinterpret_cast<int64_t *>(elementPtr);
    auto value = element.getIntegerValue();
    *elementData = (int64_t)value.getSExtValue();
    return;
  }

  // Handle unsigned integer types.
  if (elementType.isUnsignedInteger(2) || elementType.isUnsignedInteger(4) ||
      elementType.isUnsignedInteger(8)) {
    auto elementData = reinterpret_cast<uint8_t *>(elementPtr);
    auto value = element.getIntegerValue();
    *elementData = (uint8_t)value.getZExtValue();
    return;
  }

  if (elementType.isUnsignedInteger(16)) {
    auto elementData = reinterpret_cast<uint16_t *>(elementPtr);
    auto value = element.getIntegerValue();
    *elementData = (uint16_t)value.getZExtValue();
    return;
  }

  if (elementType.isUnsignedInteger(32)) {
    auto elementData = reinterpret_cast<uint32_t *>(elementPtr);
    auto value = element.getIntegerValue();
    *elementData = (uint32_t)value.getZExtValue();
    return;
  }

  if (elementType.isUnsignedInteger(64)) {
    auto elementData = reinterpret_cast<uint64_t *>(elementPtr);
    auto value = element.getIntegerValue();
    *elementData = (uint64_t)value.getZExtValue();
    return;
  }

  // Handle boolean type.
  if (isSupportedBooleanType(elementType)) {
    auto elementData = reinterpret_cast<uint8_t *>(elementPtr);
    auto value = element.getBooleanValue();
    *elementData = value ? 1 : 0;
    return;
  }

  // Handle complex types.
  if (isa<ComplexType>(elementType)) {
    auto complexElemTy = cast<ComplexType>(elementType).getElementType();
    auto complexValue = element.getComplexValue();

    if (complexElemTy.isF32()) {
      auto elementData = reinterpret_cast<std::complex<float> *>(elementPtr);
      *elementData = std::complex<float>(complexValue.real().convertToFloat(),
                                         complexValue.imag().convertToFloat());
      return;
    }

    if (complexElemTy.isF64()) {
      auto elementData = reinterpret_cast<std::complex<double> *>(elementPtr);
      *elementData =
          std::complex<double>(complexValue.real().convertToDouble(),
                               complexValue.imag().convertToDouble());
      return;
    }
  }

  report_fatal_error(invalidArgument("Unsupported element type: %s",
                                     debugString(elementType).c_str()));
}

IndexSpaceIterator Tensor::index_begin() const {
  return getShape().index_begin();
}

IndexSpaceIterator Tensor::index_end() const { return getShape().index_end(); }

namespace {

void printNewlineIndent(llvm::raw_ostream &os, int64_t n) {
  os << '\n';
  for (int64_t i = 0; i < n; ++i) os << "  ";
}

bool isEndOfIterationSpace(const Index &idx, const Sizes &shape) {
  // Check if this is the last index of the right-most dimension
  // I.e.
  //   Index{0, 3} vs Shape{0, 4, 9} ==> true
  // Since [3] is the final valid index of the second dim of shape.
  if (idx.empty()) return true;
  auto dimSize = shape[idx.size() - 1];
  return idx.back() == dimSize - 1;
}

void printHelper(llvm::raw_ostream &os, const Tensor &tensor,
                 const Sizes &shape, Index &currIdx, int64_t indent) {
  // Base case: We have a full index, print the item
  if (currIdx.size() == shape.size()) {
    os << tensor.get(currIdx);
    if (!isEndOfIterationSpace(currIdx, shape)) os << ", ";
    return;
  }

  // Recursive step: Add a dimension to currIdx and recurse.
  printNewlineIndent(os, indent);
  os << "[";
  auto currAxes = shape[currIdx.size()];
  for (int64_t idx = 0; idx < currAxes; ++idx) {
    currIdx.push_back(idx);
    printHelper(os, tensor, shape, currIdx, indent + 1);
    currIdx.pop_back();
  }
  os << "]";

  // Print separator between tensors, and print newline at end.
  if (!isEndOfIterationSpace(currIdx, shape))
    os << ",";
  else
    printNewlineIndent(os, indent - 1);
}

}  // namespace

void Tensor::print(raw_ostream &os) const {
  getType().print(os);
  os << " {";
  Index idx{};
  printHelper(os, *this, getShape(), idx, /*indent=*/1);
  os << "}";
}

void Tensor::dump() const { print(llvm::errs()); }

Tensor makeTensor(DenseElementsAttr attr) {
  auto type = attr.getType();
  auto elementType = type.getElementType();

  // Handle floating-point types.
  if (isSupportedFloatType(elementType) &&
      cast<FloatType>(elementType).getWidth() <= 8) {
    auto floatValues = llvm::map_to_vector(
        attr.getValues<APFloat>(), [&](APFloat value) -> uint8_t {
          return value.bitcastToAPInt().getZExtValue();
        });

    // For f8E3M4, f8E4M3, f8E4M3FN, f8E4M3FNUZ, f8E4M3B11FNUZ, f8E5M2, and
    // f8E5M2FNUZ, f8E8M0FNU floating-point types, we use uint8_t as their
    // storage type because there are no builtin types for those.
    // For f4E2M1FN, f6E2M3FN, and f6E3M2FN floating-point types, we still use
    // uint8_t, even though the underlying types require less bits (similar
    // to how ui2/ui4 types are handled).
    return Tensor(type, HeapAsmResourceBlob::allocateAndCopyInferAlign<uint8_t>(
                            floatValues));
  }

  if (elementType.isF16() || elementType.isBF16()) {
    auto floatValues = llvm::map_to_vector(
        attr.getValues<APFloat>(), [&](APFloat value) -> uint16_t {
          return value.bitcastToAPInt().getZExtValue();
        });

    // For both f16 and bf16 floating-point types, we use uint16_t as their
    // storage type because there are no builtin types for those.
    return Tensor(
        type,
        HeapAsmResourceBlob::allocateAndCopyInferAlign<uint16_t>(floatValues));
  }

  if (elementType.isF32()) {
    auto floatValues = llvm::map_to_vector(
        attr.getValues<APFloat>(),
        [&](APFloat value) -> float { return value.convertToFloat(); });
    return Tensor(type, HeapAsmResourceBlob::allocateAndCopyInferAlign<float>(
                            floatValues));
  }

  if (elementType.isF64()) {
    auto floatValues = llvm::map_to_vector(
        attr.getValues<APFloat>(),
        [&](APFloat value) -> double { return value.convertToDouble(); });
    return Tensor(type, HeapAsmResourceBlob::allocateAndCopyInferAlign<double>(
                            floatValues));
  }

  // Handle signed integer types.
  if (elementType.isSignlessInteger(2) || elementType.isSignlessInteger(4) ||
      elementType.isSignlessInteger(8)) {
    auto intValues = llvm::map_to_vector(
        attr.getValues<APInt>(),
        [&](APInt value) -> int8_t { return value.getSExtValue(); });
    return Tensor(type, HeapAsmResourceBlob::allocateAndCopyInferAlign<int8_t>(
                            intValues));
  }

  if (elementType.isSignlessInteger(16)) {
    auto intValues = llvm::map_to_vector(
        attr.getValues<APInt>(),
        [&](APInt value) -> int16_t { return value.getSExtValue(); });
    return Tensor(type, HeapAsmResourceBlob::allocateAndCopyInferAlign<int16_t>(
                            intValues));
  }

  if (elementType.isSignlessInteger(32)) {
    auto intValues = llvm::map_to_vector(
        attr.getValues<APInt>(),
        [&](APInt value) -> int32_t { return value.getSExtValue(); });
    return Tensor(type, HeapAsmResourceBlob::allocateAndCopyInferAlign<int32_t>(
                            intValues));
  }

  if (elementType.isSignlessInteger(64)) {
    auto intValues = llvm::map_to_vector(
        attr.getValues<APInt>(),
        [&](APInt value) -> int64_t { return value.getSExtValue(); });
    return Tensor(type, HeapAsmResourceBlob::allocateAndCopyInferAlign<int64_t>(
                            intValues));
  }

  // Handle unsigned integer types.
  if (elementType.isUnsignedInteger(2) || elementType.isUnsignedInteger(4) ||
      elementType.isUnsignedInteger(8)) {
    auto intValues = llvm::map_to_vector(
        attr.getValues<APInt>(),
        [&](APInt value) -> uint8_t { return value.getZExtValue(); });
    return Tensor(type, HeapAsmResourceBlob::allocateAndCopyInferAlign<uint8_t>(
                            intValues));
  }

  if (elementType.isUnsignedInteger(16)) {
    auto intValues = llvm::map_to_vector(
        attr.getValues<APInt>(),
        [&](APInt value) -> uint16_t { return value.getZExtValue(); });
    return Tensor(
        type,
        HeapAsmResourceBlob::allocateAndCopyInferAlign<uint16_t>(intValues));
  }

  if (elementType.isUnsignedInteger(32)) {
    auto intValues = llvm::map_to_vector(
        attr.getValues<APInt>(),
        [&](APInt value) -> uint32_t { return value.getZExtValue(); });
    return Tensor(
        type,
        HeapAsmResourceBlob::allocateAndCopyInferAlign<uint32_t>(intValues));
  }

  if (elementType.isUnsignedInteger(64)) {
    auto intValues = llvm::map_to_vector(
        attr.getValues<APInt>(),
        [&](APInt value) -> uint64_t { return value.getZExtValue(); });
    return Tensor(
        type,
        HeapAsmResourceBlob::allocateAndCopyInferAlign<uint64_t>(intValues));
  }

  // Handle boolean type.
  if (isSupportedBooleanType(elementType)) {
    auto boolValues = llvm::map_to_vector(
        attr.getValues<bool>(),
        [&](bool value) -> uint8_t { return value ? 1 : 0; });
    return Tensor(type, HeapAsmResourceBlob::allocateAndCopyInferAlign<uint8_t>(
                            boolValues));
  }

  // Handle complex types.
  if (isa<ComplexType>(elementType)) {
    auto complexElemTy = cast<ComplexType>(elementType).getElementType();
    if (complexElemTy.isF32()) {
      auto complexValues = llvm::map_to_vector(
          attr.getValues<std::complex<APFloat>>(),
          [&](std::complex<APFloat> value) -> std::complex<float> {
            return std::complex<float>(value.real().convertToFloat(),
                                       value.imag().convertToFloat());
          });
      return Tensor(
          type,
          HeapAsmResourceBlob::allocateAndCopyInferAlign<std::complex<float>>(
              complexValues));
    }
    if (complexElemTy.isF64()) {
      auto complexValues = llvm::map_to_vector(
          attr.getValues<std::complex<APFloat>>(),
          [&](std::complex<APFloat> value) -> std::complex<double> {
            return std::complex<double>(value.real().convertToDouble(),
                                        value.imag().convertToDouble());
          });
      return Tensor(
          type,
          HeapAsmResourceBlob::allocateAndCopyInferAlign<std::complex<double>>(
              complexValues));
    }
  }

  report_fatal_error(
      invalidArgument("Unsupported type: ", debugString(type).c_str()));
}

DenseElementsAttr makeDenseElementsAttr(Tensor tensor) {
  auto type = tensor.getType();
  auto elementType = type.getElementType();

  if (isa<FloatType>(elementType)) {
    std::vector<llvm::APFloat> values;
    for (auto it = tensor.index_begin(); it != tensor.index_end(); ++it) {
      Element element = tensor.get(*it);
      values.push_back(element.getFloatValue());
    }
    return DenseFPElementsAttr::get(tensor.getType(), values);
  }
  if (isa<IntegerType>(elementType)) {
    std::vector<llvm::APInt> values;
    for (auto it = tensor.index_begin(); it != tensor.index_end(); ++it) {
      Element element = tensor.get(*it);
      values.push_back(element.getIntegerValue());
    }
    return DenseIntElementsAttr::get(tensor.getType(), values);
  }

  llvm::report_fatal_error("Only FloatType and IntType are handled currently.");
}

Sizes makeSizes(Tensor tensor) {
  if (tensor.getRank() != 1 || !isa<IntegerType>(tensor.getElementType())) {
    std::string str;
    llvm::raw_string_ostream os(str);
    os << "makeSizes(Tensor) only accepts integer tensors of rank 1, but got: ";
    tensor.print(os);
    llvm::report_fatal_error(str.c_str());
  }
  SmallVector<int64_t> values;
  values.reserve(tensor.getNumElements());
  for (auto it = tensor.index_begin(), end = tensor.index_end(); it != end;
       it++) {
    values.push_back(tensor.get(*it).getIntegerValue().getSExtValue());
  }
  return Sizes(values);
}

}  // namespace stablehlo
}  // namespace mlir
