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
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "mlir/Support/DebugStringHelper.h"
#include "stablehlo/reference/Index.h"

namespace mlir {
namespace stablehlo {

namespace {

template <typename... Ts>
inline llvm::Error invalidArgument(char const *Fmt, const Ts &...Vals) {
  return createStringError(llvm::errc::invalid_argument, Fmt, Vals...);
}

int64_t getSizeInBytes(Type type) {
  if (auto shapedType = type.dyn_cast<ShapedType>())
    return shapedType.getNumElements() *
           getSizeInBytes(shapedType.getElementType());

  if (type.isIntOrFloat())
    return std::max(type.getIntOrFloatBitWidth(), (unsigned)8) / 8;

  if (auto complexType = type.dyn_cast<mlir::ComplexType>())
    return getSizeInBytes(complexType.getElementType()) * 2;

  auto err = invalidArgument("Unsupported type: %s", debugString(type).c_str());
  report_fatal_error(std::move(err));
}

// Flattens multi-dimensional index 'index' of a tensor to a linearized index
// into the underlying storage where elements are laid out in canonical order.
int64_t flattenIndex(ArrayRef<int64_t> shape, ArrayRef<int64_t> index) {
  int64_t idx = 0;
  if (shape.empty()) return idx;

  // Computes strides of a tensor shape: The number of locations in memory
  // between beginnings of successive array elements, measured in units of the
  // size of the array's elements.
  // Example: For a tensor shape [1,2,3], strides = [6,3,1]
  std::vector<int64_t> strides(shape.size());
  strides[shape.size() - 1] = 1;
  for (int i = shape.size() - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * shape[i + 1];
  }

  // Use the computed strides to flatten the multi-dimensional index 'index'
  // to a linearized index.
  // Example: For a tensor with shape [1,2,3], strides = [6,3,1], and index =
  // [0, 1, 2], the flattened index = 0*6 + 1*3 + 2*1 = 5
  for (size_t i = 0; i < index.size(); i++) {
    idx += strides[i] * index[i];
  }
  return idx;
}

bool isSupportedUnsignedIntegerType(Type type) {
  return type.isUnsignedInteger(4) || type.isUnsignedInteger(8) ||
         type.isUnsignedInteger(16) || type.isUnsignedInteger(32) ||
         type.isUnsignedInteger(64);
}

bool isSupportedSignedIntegerType(Type type) {
  // TODO(#22): StableHLO, as bootstrapped from MHLO, inherits signless
  // integers which was added in MHLO for legacy reasons. Going forward,
  // StableHLO will adopt signfull integer semantics with signed and unsigned
  // integer variants.
  return type.isSignlessInteger(4) || type.isSignlessInteger(8) ||
         type.isSignlessInteger(16) || type.isSignlessInteger(32) ||
         type.isSignlessInteger(64);
}

bool isSupportedIntegerType(Type type) {
  return isSupportedUnsignedIntegerType(type) ||
         isSupportedSignedIntegerType(type);
}

}  // namespace

namespace detail {

Buffer::Buffer(ShapedType type) : type_(type), data_(getSizeInBytes(type), 0) {}

Buffer::Buffer(ShapedType type, void *data)
    : Buffer(type, static_cast<const void *>(data)) {}

Buffer::Buffer(ShapedType type, const void *data)
    : type_(type),
      data_(static_cast<const char *>(data),
            static_cast<const char *>(data) + getSizeInBytes(type)) {}

}  // namespace detail

Tensor::Tensor() {}

Tensor::Tensor(ShapedType type)
    : impl_(llvm::makeIntrusiveRefCnt<detail::Buffer>(type)) {}

Tensor::Tensor(DenseElementsAttr attr) {
  // TODO(sdasgup3): We're using DenseElementsAttr::getRawData() here for
  // simplicity, because it provides a contiguous representation of underlying
  // data in most cases. However, this doesn't always work (e.g. for splat or
  // for i1), so we'll be migrating to something more reliable in the near
  // future.
  impl_ = llvm::makeIntrusiveRefCnt<detail::Buffer>(attr.getType(),
                                                    attr.getRawData().data());
}

ShapedType Tensor::getType() const { return impl_->getType(); }

int64_t Tensor::getNumElements() const { return getType().getNumElements(); }

Element Tensor::get(ArrayRef<int64_t> index) const {
  Type elementType = getType().getElementType();
  char *elementPtr =
      impl_->getData() +
      getSizeInBytes(elementType) * flattenIndex(getType().getShape(), index);

  // Handle floating-point types.
  if (elementType.isF16()) {
    auto elementData = reinterpret_cast<uint16_t *>(elementPtr);
    return Element(elementType, APFloat(llvm::APFloatBase::IEEEhalf(),
                                        APInt(16, *elementData)));
  }

  if (elementType.isBF16()) {
    auto elementData = reinterpret_cast<uint16_t *>(elementPtr);
    return Element(elementType, APFloat(llvm::APFloatBase::BFloat(),
                                        APInt(16, *elementData)));
  }

  if (elementType.isF32()) {
    auto elementData = reinterpret_cast<float *>(elementPtr);
    return Element(elementType, APFloat(*elementData));
  }

  if (elementType.isF64()) {
    auto elementData = reinterpret_cast<double *>(elementPtr);
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
      auto elementData = reinterpret_cast<int8_t *>(elementPtr);
      return Element(elementType, APInt(intTy.getWidth(), *elementData,
                                        intTy.isSignedInteger()));
    } else if (elementType.isSignlessInteger(16)) {
      auto elementData = reinterpret_cast<int16_t *>(elementPtr);
      return Element(elementType, APInt(intTy.getWidth(), *elementData,
                                        intTy.isSignedInteger()));
    } else if (elementType.isSignlessInteger(32)) {
      auto elementData = reinterpret_cast<int32_t *>(elementPtr);
      return Element(elementType, APInt(intTy.getWidth(), *elementData,
                                        intTy.isSignedInteger()));
    } else if (elementType.isSignlessInteger(64)) {
      auto elementData = reinterpret_cast<int64_t *>(elementPtr);
      return Element(elementType, APInt(intTy.getWidth(), *elementData,
                                        intTy.isSignedInteger()));
    } else if (elementType.isUnsignedInteger(4) ||
               elementType.isUnsignedInteger(8)) {
      auto elementData = reinterpret_cast<uint8_t *>(elementPtr);
      return Element(elementType, APInt(intTy.getWidth(), *elementData,
                                        intTy.isSignedInteger()));
    } else if (elementType.isUnsignedInteger(16)) {
      auto elementData = reinterpret_cast<uint16_t *>(elementPtr);
      return Element(elementType, APInt(intTy.getWidth(), *elementData,
                                        intTy.isSignedInteger()));
    } else if (elementType.isUnsignedInteger(32)) {
      auto elementData = reinterpret_cast<uint32_t *>(elementPtr);
      return Element(elementType, APInt(intTy.getWidth(), *elementData,
                                        intTy.isSignedInteger()));
    } else if (elementType.isUnsignedInteger(64)) {
      auto elementData = reinterpret_cast<uint64_t *>(elementPtr);
      return Element(elementType, APInt(intTy.getWidth(), *elementData,
                                        intTy.isSignedInteger()));
    }
  }

  // Handle complex types.
  if (elementType.isa<ComplexType>()) {
    auto complexElemTy = elementType.cast<ComplexType>().getElementType();

    if (complexElemTy.isF32()) {
      auto elementData = reinterpret_cast<std::complex<float> *>(elementPtr);
      return Element(elementType,
                     std::complex<APFloat>(APFloat(elementData->real()),
                                           APFloat(elementData->imag())));
    }

    if (complexElemTy.isF64()) {
      auto elementData = reinterpret_cast<std::complex<double> *>(elementPtr);
      return Element(elementType,
                     std::complex<APFloat>(APFloat(elementData->real()),
                                           APFloat(elementData->imag())));
    }
  }

  auto err = invalidArgument("Unsupported element type: %s",
                             debugString(elementType).c_str());
  report_fatal_error(std::move(err));
}

void Tensor::set(ArrayRef<int64_t> index, const Element &element) {
  Type elementType = getType().getElementType();
  char *elementPtr =
      impl_->getData() +
      getSizeInBytes(elementType) * flattenIndex(getType().getShape(), index);

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

  auto err = invalidArgument("Unsupported element type: %s",
                             debugString(elementType).c_str());
  report_fatal_error(std::move(err));
}

IndexSpaceIterator Tensor::index_begin() const {
  auto shape = getType().getShape();
  SmallVector<int64_t> index(shape.size());
  return IndexSpaceIterator(shape, index);
}

IndexSpaceIterator Tensor::index_end() const {
  auto shape = getType().getShape();
  return IndexSpaceIterator(shape, {});
}

void Tensor::print(raw_ostream &os) const {
  getType().print(os);
  os << " {\n";

  for (auto it = this->index_begin(); it != this->index_end(); ++it)
    os << "  " << get(*it) << "\n";

  os << "}";
}

void Tensor::dump() const {
  print(llvm::errs());
  llvm::errs() << "\n";
}

Tensor makeTensor(ShapedType type, ArrayRef<StringRef> strData) {
  auto elemType = type.getElementType();

  // We are not using parseAttribute for parsing Float literals mainly because
  // it does not parse special float values like nan, +/-inf.
  if (auto complexTy = elemType.dyn_cast<ComplexType>()) {
    auto complexElemTy = complexTy.getElementType();
    auto floatType = complexElemTy.dyn_cast<FloatType>();
    if (!floatType) {
      auto err = invalidArgument("Unsupported element type %s for complex type",
                                 debugString(complexElemTy).c_str());
      report_fatal_error(std::move(err));
    }

    auto floatValues = llvm::to_vector(
        llvm::map_range(strData, [&](StringRef strNum) -> APFloat {
          return APFloat(floatType.getFloatSemantics(), strNum);
        }));

    auto complexData = llvm::makeArrayRef(
        reinterpret_cast<std::complex<APFloat> *>(floatValues.data()),
        floatValues.size() / 2);
    return Tensor(DenseElementsAttr::get(type, complexData));
  }

  if (auto floatType = elemType.dyn_cast<FloatType>()) {
    auto floatValues =
        llvm::to_vector(llvm::map_range(strData, [&](StringRef str) -> APFloat {
          return APFloat(floatType.getFloatSemantics(), str);
        }));

    return Tensor(DenseElementsAttr::get(type, floatValues));
  }

  if (elemType.isa<IntegerType>()) {
    SmallVector<APInt> intValues;
    intValues = llvm::to_vector(
        llvm::map_range(strData, [elemType](StringRef str) -> APInt {
          return APInt(elemType.getIntOrFloatBitWidth(), str, 10);
        }));
    return Tensor(DenseElementsAttr::get(type, intValues));
  }

  auto err =
      invalidArgument("Unsupported type: %s", debugString(elemType).c_str());
  report_fatal_error(std::move(err));
}

}  // namespace stablehlo
}  // namespace mlir
