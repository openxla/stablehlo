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

#include <complex>

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/Support/Error.h"
#include "mlir/Support/DebugStringHelper.h"
#include "stablehlo/reference/Errors.h"
#include "stablehlo/reference/Types.h"

namespace mlir {
namespace stablehlo {

namespace {

int64_t getSizeInBytes(Type type) {
  if (auto shapedType = type.dyn_cast<ShapedType>())
    return shapedType.getNumElements() *
           getSizeInBytes(shapedType.getElementType());

  if (type.isIntOrFloat())
    return std::max(type.getIntOrFloatBitWidth(), (unsigned)8) / 8;

  if (auto complexType = type.dyn_cast<mlir::ComplexType>())
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

Buffer::Buffer(TensorType type)
    : type_(type),
      blob_(
          HeapAsmResourceBlob::allocate(getSizeInBytes(type), alignof(char))) {}

Buffer::Buffer(TensorType type, AsmResourceBlob blob)
    : type_(type), blob_(std::move(blob)) {}

}  // namespace detail

Tensor::Tensor() {}

Tensor::Tensor(TensorType type)
    : impl_(llvm::makeIntrusiveRefCnt<detail::Buffer>(type)) {}

Tensor::Tensor(TensorType type, AsmResourceBlob blob)
    : impl_(llvm::makeIntrusiveRefCnt<detail::Buffer>(type, std::move(blob))) {}

Element Tensor::get(const Index &index) const {
  Type elementType = getType().getElementType();
  const char *elementPtr =
      impl_->getData().data() +
      getSizeInBytes(elementType) * flattenIndex(getShape(), index);

  // Handle floating-point types.
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
    IntegerType intTy = elementType.cast<IntegerType>();

    if (elementType.isSignlessInteger(4) || elementType.isSignlessInteger(8)) {
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
    } else if (elementType.isUnsignedInteger(4) ||
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
  if (elementType.isa<ComplexType>()) {
    auto complexElemTy = elementType.cast<ComplexType>().getElementType();

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
  if (elementType.isSignlessInteger(4) || elementType.isSignlessInteger(8)) {
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
  if (elementType.isUnsignedInteger(4) || elementType.isUnsignedInteger(8)) {
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
  if (elementType.isa<ComplexType>()) {
    auto complexElemTy = elementType.cast<ComplexType>().getElementType();
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
  auto shape = getShape();

  if (any_of(shape, [](int64_t dimSize) { return dimSize == 0; }))
    return IndexSpaceIterator(shape, std::nullopt);

  Index initialIndex(shape.size());
  return IndexSpaceIterator(shape, initialIndex);
}

IndexSpaceIterator Tensor::index_end() const {
  return IndexSpaceIterator(getShape(), std::nullopt);
}

void Tensor::print(raw_ostream &os) const {
  getType().print(os);
  os << " {\n";

  for (auto it = this->index_begin(); it != this->index_end(); ++it)
    os << "  " << get(*it) << "\n";

  os << "}";
}

void Tensor::dump() const { print(llvm::errs()); }

Tensor makeTensor(DenseElementsAttr attr) {
  auto type = attr.getType().cast<TensorType>();
  auto elemType = type.getElementType();

  // Handle floating-point types.
  if (elemType.isF16() || elemType.isBF16()) {
    auto floatValues = llvm::to_vector(llvm::map_range(
        attr.getValues<APFloat>(), [&](APFloat value) -> uint16_t {
          return value.bitcastToAPInt().getZExtValue();
        }));

    // For both f16 and bf16 floating-point types, we use uint16_t as their
    // storage type because there are no buitin types for those.
    return Tensor(
        type,
        HeapAsmResourceBlob::allocateAndCopyInferAlign<uint16_t>(floatValues));
  }

  if (elemType.isF32()) {
    auto floatValues = llvm::to_vector(llvm::map_range(
        attr.getValues<APFloat>(),
        [&](APFloat value) -> float { return value.convertToFloat(); }));
    return Tensor(type, HeapAsmResourceBlob::allocateAndCopyInferAlign<float>(
                            floatValues));
  }

  if (elemType.isF64()) {
    auto floatValues = llvm::to_vector(llvm::map_range(
        attr.getValues<APFloat>(),
        [&](APFloat value) -> double { return value.convertToDouble(); }));
    return Tensor(type, HeapAsmResourceBlob::allocateAndCopyInferAlign<double>(
                            floatValues));
  }

  // Handle signed integer types.
  if (elemType.isSignlessInteger(4) || elemType.isSignlessInteger(8)) {
    auto intValues = llvm::to_vector(llvm::map_range(
        attr.getValues<APInt>(),
        [&](APInt value) -> int8_t { return value.getSExtValue(); }));
    return Tensor(type, HeapAsmResourceBlob::allocateAndCopyInferAlign<int8_t>(
                            intValues));
  }

  if (elemType.isSignlessInteger(16)) {
    auto intValues = llvm::to_vector(llvm::map_range(
        attr.getValues<APInt>(),
        [&](APInt value) -> int16_t { return value.getSExtValue(); }));
    return Tensor(type, HeapAsmResourceBlob::allocateAndCopyInferAlign<int16_t>(
                            intValues));
  }

  if (elemType.isSignlessInteger(32)) {
    auto intValues = llvm::to_vector(llvm::map_range(
        attr.getValues<APInt>(),
        [&](APInt value) -> int32_t { return value.getSExtValue(); }));
    return Tensor(type, HeapAsmResourceBlob::allocateAndCopyInferAlign<int32_t>(
                            intValues));
  }

  if (elemType.isSignlessInteger(64)) {
    auto intValues = llvm::to_vector(llvm::map_range(
        attr.getValues<APInt>(),
        [&](APInt value) -> int64_t { return value.getSExtValue(); }));
    return Tensor(type, HeapAsmResourceBlob::allocateAndCopyInferAlign<int64_t>(
                            intValues));
  }

  // Handle unsigned integer types.
  if (elemType.isUnsignedInteger(4) || elemType.isUnsignedInteger(8)) {
    auto intValues = llvm::to_vector(llvm::map_range(
        attr.getValues<APInt>(),
        [&](APInt value) -> uint8_t { return value.getZExtValue(); }));
    return Tensor(type, HeapAsmResourceBlob::allocateAndCopyInferAlign<uint8_t>(
                            intValues));
  }

  if (elemType.isUnsignedInteger(16)) {
    auto intValues = llvm::to_vector(llvm::map_range(
        attr.getValues<APInt>(),
        [&](APInt value) -> uint16_t { return value.getZExtValue(); }));
    return Tensor(
        type,
        HeapAsmResourceBlob::allocateAndCopyInferAlign<uint16_t>(intValues));
  }

  if (elemType.isUnsignedInteger(32)) {
    auto intValues = llvm::to_vector(llvm::map_range(
        attr.getValues<APInt>(),
        [&](APInt value) -> uint32_t { return value.getZExtValue(); }));
    return Tensor(
        type,
        HeapAsmResourceBlob::allocateAndCopyInferAlign<uint32_t>(intValues));
  }

  if (elemType.isUnsignedInteger(64)) {
    auto intValues = llvm::to_vector(llvm::map_range(
        attr.getValues<APInt>(),
        [&](APInt value) -> uint64_t { return value.getZExtValue(); }));
    return Tensor(
        type,
        HeapAsmResourceBlob::allocateAndCopyInferAlign<uint64_t>(intValues));
  }

  // Handle boolean type.
  if (isSupportedBooleanType(elemType)) {
    auto boolValues = llvm::to_vector(
        llvm::map_range(attr.getValues<bool>(),
                        [&](bool value) -> uint8_t { return value ? 1 : 0; }));
    return Tensor(type, HeapAsmResourceBlob::allocateAndCopyInferAlign<uint8_t>(
                            boolValues));
  }

  // Handle complex types.
  if (elemType.isa<ComplexType>()) {
    auto complexElemTy = elemType.cast<ComplexType>().getElementType();
    if (complexElemTy.isF32()) {
      auto complexValues = llvm::to_vector(llvm::map_range(
          attr.getValues<std::complex<APFloat>>(),
          [&](std::complex<APFloat> value) -> std::complex<float> {
            return std::complex<float>(value.real().convertToFloat(),
                                       value.imag().convertToFloat());
          }));
      return Tensor(
          type,
          HeapAsmResourceBlob::allocateAndCopyInferAlign<std::complex<float>>(
              complexValues));
    }
    if (complexElemTy.isF64()) {
      auto complexValues = llvm::to_vector(llvm::map_range(
          attr.getValues<std::complex<APFloat>>(),
          [&](std::complex<APFloat> value) -> std::complex<double> {
            return std::complex<double>(value.real().convertToDouble(),
                                        value.imag().convertToDouble());
          }));
      return Tensor(
          type,
          HeapAsmResourceBlob::allocateAndCopyInferAlign<std::complex<double>>(
              complexValues));
    }
  }

  report_fatal_error(
      invalidArgument("Unsupported type: ", debugString(type).c_str()));
}

}  // namespace stablehlo
}  // namespace mlir
