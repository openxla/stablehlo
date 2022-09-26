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

template <typename IntegerFn, typename FloatFn, typename ComplexFn>
Element map(const Element &el, IntegerFn integerFn, FloatFn floatFn,
            ComplexFn complexFn) {
  Type type = el.getType();

  if (isSupportedIntegerType(type)) {
    auto intEl = getIntegerValue(el);
    return Element(type, IntegerAttr::get(type, integerFn(intEl)));
  }

  if (isSupportedFloatType(type)) {
    auto floatEl = getFloatValue(el);
    return Element(type, FloatAttr::get(type, floatFn(floatEl)));
  }

  if (isSupportedComplexType(type)) {
    auto complexElemTy = type.cast<ComplexType>().getElementType();
    auto complexEl = getComplexValue(el);
    auto complexResult = complexFn(complexEl);

    return Element(
        type,
        ArrayAttr::get(type.getContext(),
                       {FloatAttr::get(complexElemTy, complexResult.real()),
                        FloatAttr::get(complexElemTy, complexResult.imag())}));
  }

  report_fatal_error(invalidArgument("Unsupported element type: %s",
                                     debugString(type).c_str()));
}

template <typename IntegerFn, typename FloatFn, typename ComplexFn>
Element map(const Element &lhs, const Element &rhs, IntegerFn integerFn,
            FloatFn floatFn, ComplexFn complexFn) {
  Type type = lhs.getType();
  if (lhs.getType() != rhs.getType())
    report_fatal_error(invalidArgument("Element types don't match: %s vs %s",
                                       debugString(lhs.getType()).c_str(),
                                       debugString(rhs.getType()).c_str()));

  if (isSupportedIntegerType(type)) {
    auto intLhs = getIntegerValue(lhs);
    auto intRhs = getIntegerValue(rhs);
    return Element(type, IntegerAttr::get(type, integerFn(intLhs, intRhs)));
  }

  if (isSupportedFloatType(type)) {
    auto floatLhs = getFloatValue(lhs);
    auto floatRhs = getFloatValue(rhs);
    return Element(type, FloatAttr::get(type, floatFn(floatLhs, floatRhs)));
  }

  if (isSupportedComplexType(type)) {
    auto complexElemTy = type.cast<ComplexType>().getElementType();
    auto complexLhs = getComplexValue(lhs);
    auto complexRhs = getComplexValue(rhs);
    auto complexResult = complexFn(complexLhs, complexRhs);

    return Element(
        type,
        ArrayAttr::get(type.getContext(),
                       {FloatAttr::get(complexElemTy, complexResult.real()),
                        FloatAttr::get(complexElemTy, complexResult.imag())}));
  }

  report_fatal_error(invalidArgument("Unsupported element type: %s",
                                     debugString(type).c_str()));
}

template <typename FloatFn, typename ComplexFn>
Element mapWithUpcastToDouble(const Element &el, FloatFn floatFn,
                              ComplexFn complexFn) {
  Type type = el.getType();

  if (isSupportedFloatType(type)) {
    APFloat elVal = getFloatValue(el);
    const llvm::fltSemantics &elSemantics = elVal.getSemantics();
    APFloat resultVal(floatFn(elVal.convertToDouble()));
    bool roundingErr;
    resultVal.convert(elSemantics, APFloat::rmNearestTiesToEven, &roundingErr);
    return Element(type, FloatAttr::get(type, resultVal));
  }

  if (isSupportedComplexType(type)) {
    Type elType = type.cast<ComplexType>().getElementType();
    auto elVal = getComplexValue(el);
    const llvm::fltSemantics &elSemantics = elVal.real().getSemantics();
    auto resultVal = complexFn(std::complex<double>(
        elVal.real().convertToDouble(), elVal.imag().convertToDouble()));
    bool roundingErr;
    APFloat resultReal(resultVal.real());
    resultReal.convert(elSemantics, APFloat::rmNearestTiesToEven, &roundingErr);
    APFloat resultImag(resultVal.imag());
    resultImag.convert(elSemantics, APFloat::rmNearestTiesToEven, &roundingErr);
    return Element(type, ArrayAttr::get(type.getContext(),
                                        {FloatAttr::get(elType, resultReal),
                                         FloatAttr::get(elType, resultImag)}));
  }

  report_fatal_error(invalidArgument("Unsupported element type: %s",
                                     debugString(type).c_str()));
}

}  // namespace

Element Element::operator+(const Element &other) const {
  return map(
      *this, other, [](APInt lhs, APInt rhs) { return lhs + rhs; },
      [](APFloat lhs, APFloat rhs) { return lhs + rhs; },
      [](std::complex<APFloat> lhs, std::complex<APFloat> rhs) {
        // NOTE: lhs + rhs doesn't work for std::complex<APFloat>
        // because the default implementation for the std::complex template
        // needs operator+= which is not defined on APFloat.
        auto resultReal = lhs.real() + rhs.real();
        auto resultImag = lhs.imag() + rhs.imag();
        return std::complex<APFloat>(resultReal, resultImag);
      });
}

Element sine(const Element &e) {
  return mapWithUpcastToDouble(
      e, [](double e) { return std::sin(e); },
      [](std::complex<double> e) { return std::sin(e); });
}

void Element::print(raw_ostream &os) const { value_.print(os); }

void Element::dump() const {
  print(llvm::errs());
  llvm::errs() << "\n";
}

}  // namespace stablehlo
}  // namespace mlir
