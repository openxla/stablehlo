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

#include "stablehlo/reference/Element.h"

#include <complex>

#include "llvm/ADT/APFloat.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/DebugStringHelper.h"

namespace mlir {
namespace stablehlo {

namespace {

template <typename... Ts>
inline llvm::Error invalidArgument(char const *Fmt, const Ts &...Vals) {
  return createStringError(llvm::errc::invalid_argument, Fmt, Vals...);
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

bool isSupportedFloatType(Type type) {
  return type.isF16() || type.isBF16() || type.isF32() || type.isF64();
}

bool isSupportedComplexType(Type type) {
  auto complexTy = type.dyn_cast<mlir::ComplexType>();
  if (!complexTy) return false;

  auto complexElemTy = complexTy.getElementType();
  return complexElemTy.isF32() || complexElemTy.isF64();
}

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

Element Element::operator+(const Element &other) const {
  auto type = getType();
  assert(type == other.getType());

  if (isSupportedFloatType(type)) {
    auto left = getFloatValue(*this);
    auto right = getFloatValue(other);
    return Element(type, FloatAttr::get(type, left + right));
  }

  if (isSupportedIntegerType(type)) {
    auto left = getIntegerValue(*this);
    auto right = getIntegerValue(other);
    return Element(type, IntegerAttr::get(type, left + right));
  }

  if (isSupportedComplexType(type)) {
    auto complexElemTy = type.cast<ComplexType>().getElementType();
    auto leftComplexValue = getComplexValue(*this);
    auto rightComplexValue = getComplexValue(other);

    return Element(
        type,
        ArrayAttr::get(
            type.getContext(),
            {FloatAttr::get(complexElemTy,
                            leftComplexValue.real() + rightComplexValue.real()),
             FloatAttr::get(complexElemTy, leftComplexValue.imag() +
                                               rightComplexValue.imag())}));
  }

  // Report error.
  auto err = invalidArgument("Unsupported element type: %s",
                             debugString(type).c_str());
  report_fatal_error(std::move(err));
}

Element sine(const Element &e) {
  Type type = e.getType();
  if (isSupportedComplexType(type)) {
    auto complex = getComplexValue(e);
    APFloat realAPF = complex.real();
    APFloat imagAPF = complex.imag();
    Type elementType = e.getValue()
                           .cast<ArrayAttr>()
                           .getValue()[0]
                           .cast<FloatAttr>()
                           .getType();
    if (elementType.isF32()) {
      std::complex<float> complex(realAPF.convertToFloat(),
                                  imagAPF.convertToFloat());
      std::complex<float> sinVal = std::sin(complex);
      return Element(
          elementType,
          ArrayAttr::get(
              elementType.getContext(),
              {FloatAttr::get(elementType, APFloat(sinVal.real())),
               FloatAttr::get(elementType, APFloat(sinVal.imag()))}));
    } else if (elementType.isF64()) {
      std::complex<double> complex(realAPF.convertToDouble(),
                                   imagAPF.convertToDouble());
      std::complex<double> sinVal = std::sin(complex);
      return Element(
          elementType,
          ArrayAttr::get(
              elementType.getContext(),
              {FloatAttr::get(elementType, APFloat(sinVal.real())),
               FloatAttr::get(elementType, APFloat(sinVal.imag()))}));
    }
  } else if (isSupportedFloatType(type)) {
    APFloat val = e.getValue().cast<FloatAttr>().getValue();
    if (type.isBF16()) {
      APFloat sinVal = APFloat(std::sin(val.convertToFloat()));
      bool roundingErr;
      sinVal.convert(APFloat::BFloat(), llvm::RoundingMode::NearestTiesToEven,
                     &roundingErr);
      return Element(type, FloatAttr::get(type, sinVal));
    } else if (type.isF16()) {
      APFloat sinVal = APFloat(std::sin(val.convertToFloat()));
      bool roundingErr;
      sinVal.convert(APFloat::IEEEhalf(), llvm::RoundingMode::NearestTiesToEven,
                     &roundingErr);
      return Element(type, FloatAttr::get(type, sinVal));
    } else if (type.isF32()) {
      float sinVal = std::sin(val.convertToFloat());
      return Element(type, FloatAttr::get(type, APFloat(sinVal)));
    } else if (type.isF64()) {
      double sinVal = std::sin(val.convertToDouble());
      return Element(type, FloatAttr::get(type, APFloat(sinVal)));
    }
  }

  // Report error.
  auto err = invalidArgument("Unsupported element type: %s",
                             debugString(type).c_str());
  report_fatal_error(std::move(err));
}

void Element::print(raw_ostream &os) const { value_.print(os); }

void Element::dump() const {
  print(llvm::errs());
  llvm::errs() << "\n";
}

}  // namespace stablehlo
}  // namespace mlir
