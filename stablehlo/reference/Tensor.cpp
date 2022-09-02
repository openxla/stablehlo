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

#include "llvm/ADT/APFloat.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/DebugStringHelper.h"

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

Element Tensor::get(int64_t index) const {
  Type elementType = getType().getElementType();
  char *elementPtr = impl_->getData() + getSizeInBytes(elementType) * index;

  // Handle floating-point types.
  if (elementType.isF16()) {
    auto elementData = reinterpret_cast<uint16_t *>(elementPtr);
    auto value =
        APFloat(llvm::APFloatBase::IEEEhalf(), APInt(16, *elementData));
    return Element(elementType, FloatAttr::get(elementType, value));
  }

  if (elementType.isBF16()) {
    auto elementData = reinterpret_cast<uint16_t *>(elementPtr);
    auto value = APFloat(llvm::APFloatBase::BFloat(), APInt(16, *elementData));
    return Element(elementType, FloatAttr::get(elementType, value));
  }

  if (elementType.isF32()) {
    auto elementData = reinterpret_cast<float *>(elementPtr);
    return Element(elementType,
                   FloatAttr::get(elementType, APFloat(*elementData)));
  }

  if (elementType.isF64()) {
    auto elementData = reinterpret_cast<double *>(elementPtr);
    return Element(elementType,
                   FloatAttr::get(elementType, APFloat(*elementData)));
  }

  // Handle signed integer types.
  // TODO(#22): StableHLO, as bootstrapped from MHLO, inherits signless
  // integers which was added in MHLO for legacy reasons. Going forward,
  // StableHLO will adopt signfull integer semantics with signed and unsigned
  // integer variants.
  if (elementType.isSignlessInteger(4) || elementType.isSignlessInteger(8)) {
    auto elementData = reinterpret_cast<int8_t *>(elementPtr);
    return Element(elementType, IntegerAttr::get(elementType, *elementData));
  }

  if (elementType.isSignlessInteger(16)) {
    auto elementData = reinterpret_cast<int16_t *>(elementPtr);
    return Element(elementType, IntegerAttr::get(elementType, *elementData));
  }

  if (elementType.isSignlessInteger(32)) {
    auto elementData = reinterpret_cast<int32_t *>(elementPtr);
    return Element(elementType, IntegerAttr::get(elementType, *elementData));
  }

  if (elementType.isSignlessInteger(64)) {
    auto elementData = reinterpret_cast<int64_t *>(elementPtr);
    return Element(elementType, IntegerAttr::get(elementType, *elementData));
  }

  // Handle unsigned integer types.
  if (elementType.isUnsignedInteger(4) || elementType.isUnsignedInteger(8)) {
    auto elementData = reinterpret_cast<uint8_t *>(elementPtr);
    return Element(elementType, IntegerAttr::get(elementType, *elementData));
  }

  if (elementType.isUnsignedInteger(16)) {
    auto elementData = reinterpret_cast<uint16_t *>(elementPtr);
    return Element(elementType, IntegerAttr::get(elementType, *elementData));
  }

  if (elementType.isUnsignedInteger(32)) {
    auto elementData = reinterpret_cast<uint32_t *>(elementPtr);
    return Element(elementType, IntegerAttr::get(elementType, *elementData));
  }

  if (elementType.isUnsignedInteger(64)) {
    auto elementData = reinterpret_cast<uint64_t *>(elementPtr);
    return Element(elementType, IntegerAttr::get(elementType, *elementData));
  }

  // Handle complex types.
  if (elementType.isa<ComplexType>()) {
    auto complexElemTy = elementType.cast<ComplexType>().getElementType();

    if (complexElemTy.isF32()) {
      auto elementData = reinterpret_cast<std::complex<float> *>(elementPtr);
      return Element(
          elementType,
          ArrayAttr::get(
              elementType.getContext(),
              {FloatAttr::get(complexElemTy, APFloat(elementData->real())),
               FloatAttr::get(complexElemTy, APFloat(elementData->imag()))}));
    }

    if (complexElemTy.isF64()) {
      auto elementData = reinterpret_cast<std::complex<double> *>(elementPtr);
      return Element(
          elementType,
          ArrayAttr::get(
              elementType.getContext(),
              {FloatAttr::get(complexElemTy, APFloat(elementData->real())),
               FloatAttr::get(complexElemTy, APFloat(elementData->imag()))}));
    }
  }

  auto err = invalidArgument("Unsupported element type: %s",
                             debugString(elementType).c_str());
  report_fatal_error(std::move(err));
}

namespace {

APFloat getFloatValue(Element element) {
  return element.getValue().cast<FloatAttr>().getValue();
}

APInt getIntegerValue(Element element) {
  return element.getValue().cast<IntegerAttr>().getValue();
}

std::complex<APFloat> getComplexValue(Element element) {
  auto arryOfAttr = element.getValue().cast<ArrayAttr>().getValue();
  return std::complex<APFloat>(arryOfAttr[0].cast<FloatAttr>().getValue(),
                               arryOfAttr[1].cast<FloatAttr>().getValue());
}

}  // namespace

void Tensor::set(int64_t index, Element element) {
  Type elementType = getType().getElementType();
  char *elementPtr = impl_->getData() + getSizeInBytes(elementType) * index;

  // Handle floating-point types.
  if (elementType.isF16() || elementType.isBF16()) {
    auto elementData = reinterpret_cast<uint16_t *>(elementPtr);
    auto value = getFloatValue(element);
    *elementData = (uint16_t)value.bitcastToAPInt().getZExtValue();
    return;
  }

  if (elementType.isF32()) {
    auto elementData = reinterpret_cast<float *>(elementPtr);
    auto value = getFloatValue(element);
    *elementData = value.convertToFloat();
    return;
  }

  if (elementType.isF64()) {
    auto elementData = reinterpret_cast<double *>(elementPtr);
    auto value = getFloatValue(element);
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
    auto value = getIntegerValue(element);
    *elementData = (int8_t)value.getSExtValue();
    return;
  }

  if (elementType.isSignlessInteger(16)) {
    auto elementData = reinterpret_cast<int16_t *>(
        impl_->getData() + getSizeInBytes(elementType) * index);
    auto value = getIntegerValue(element);
    *elementData = (int16_t)value.getSExtValue();
    return;
  }

  if (elementType.isSignlessInteger(32)) {
    auto elementData = reinterpret_cast<int32_t *>(elementPtr);
    auto value = getIntegerValue(element);
    *elementData = (int32_t)value.getSExtValue();
    return;
  }

  if (elementType.isSignlessInteger(64)) {
    auto elementData = reinterpret_cast<int64_t *>(elementPtr);
    auto value = getIntegerValue(element);
    *elementData = (int64_t)value.getSExtValue();
    return;
  }

  // Handle unsigned integer types.
  if (elementType.isUnsignedInteger(4) || elementType.isUnsignedInteger(8)) {
    auto elementData = reinterpret_cast<uint8_t *>(elementPtr);
    auto value = getIntegerValue(element);
    *elementData = (uint8_t)value.getZExtValue();
    return;
  }

  if (elementType.isUnsignedInteger(16)) {
    auto elementData = reinterpret_cast<uint16_t *>(elementPtr);
    auto value = getIntegerValue(element);
    *elementData = (uint16_t)value.getZExtValue();
    return;
  }

  if (elementType.isUnsignedInteger(32)) {
    auto elementData = reinterpret_cast<uint32_t *>(elementPtr);
    auto value = getIntegerValue(element);
    *elementData = (uint32_t)value.getZExtValue();
    return;
  }

  if (elementType.isUnsignedInteger(64)) {
    auto elementData = reinterpret_cast<uint64_t *>(elementPtr);
    auto value = getIntegerValue(element);
    *elementData = (uint64_t)value.getZExtValue();
    return;
  }

  // Handle complex types.
  if (elementType.isa<ComplexType>()) {
    auto complexElemTy = elementType.cast<ComplexType>().getElementType();

    if (complexElemTy.isF32()) {
      auto elementData = reinterpret_cast<std::complex<float> *>(elementPtr);
      auto complexValue = getComplexValue(element);
      *elementData = std::complex<float>(complexValue.real().convertToFloat(),
                                         complexValue.imag().convertToFloat());
      return;
    }

    if (complexElemTy.isF64()) {
      auto elementData = reinterpret_cast<std::complex<double> *>(elementPtr);
      auto complexValue = getComplexValue(element);
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

void Tensor::print(raw_ostream &os) const {
  getType().print(os);
  os << " {\n";
  for (auto i = 0; i < getNumElements(); ++i) {
    os << "  " << get(i) << "\n";
  }
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
