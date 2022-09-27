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
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Support/DebugStringHelper.h"
#include "stablehlo/reference/Types.h"

namespace mlir {
namespace stablehlo {

namespace {

template <typename... Ts>
inline llvm::Error invalidArgument(char const *Fmt, const Ts &...Vals) {
  return createStringError(llvm::errc::invalid_argument, Fmt, Vals...);
}

template <typename IntegerFn, typename FloatFn, typename ComplexFn>
Element map(const Element &el, IntegerFn integerFn, FloatFn floatFn,
            ComplexFn complexFn) {
  Type type = el.getType();

  if (isSupportedIntegerType(type)) {
    auto intEl = el.getIntegerValue();
    return Element(type, integerFn(intEl));
  }

  if (isSupportedFloatType(type)) {
    auto floatEl = el.getFloatValue();
    return Element(type, floatFn(floatEl));
  }

  if (isSupportedComplexType(type)) {
    auto complexEl = el.getComplexValue();
    auto complexResult = complexFn(complexEl);
    return Element(type, complexResult);
  }

  report_fatal_error(invalidArgument("Unsupported element type: %s",
                                     debugString(type).c_str()));
}

template <typename SIntegerFn, typename UIntegerFn, typename FloatFn,
          typename ComplexFn>
Element map(const Element &lhs, const Element &rhs, SIntegerFn signedIntegerFn,
            UIntegerFn unsignedIntegerFn, FloatFn floatFn,
            ComplexFn complexFn) {
  Type type = lhs.getType();
  if (lhs.getType() != rhs.getType())
    report_fatal_error(invalidArgument("Element types don't match: %s vs %s",
                                       debugString(lhs.getType()).c_str(),
                                       debugString(rhs.getType()).c_str()));

<<<<<<< HEAD
  if (isSupportedIntegerType(type)) {
    auto intLhs = lhs.getIntegerValue();
    auto intRhs = rhs.getIntegerValue();
    return Element(type, integerFn(intLhs, intRhs));
=======
  if (isSupportedSignedIntegerType(type)) {
    auto intLhs = getIntegerValue(lhs);
    auto intRhs = getIntegerValue(rhs);
    return Element(type,
                   IntegerAttr::get(type, signedIntegerFn(intLhs, intRhs)));
  }

  if (isSupportedUnsignedIntegerType(type)) {
    auto intLhs = getIntegerValue(lhs);
    auto intRhs = getIntegerValue(rhs);
    return Element(type,
                   IntegerAttr::get(type, unsignedIntegerFn(intLhs, intRhs)));
>>>>>>> 6b2f370 (Use map to implment the max/min as per review comments)
  }

  if (isSupportedFloatType(type)) {
    auto floatLhs = lhs.getFloatValue();
    auto floatRhs = rhs.getFloatValue();
    return Element(type, floatFn(floatLhs, floatRhs));
  }

  if (isSupportedComplexType(type)) {
    auto complexLhs = lhs.getComplexValue();
    auto complexRhs = rhs.getComplexValue();
    auto complexResult = complexFn(complexLhs, complexRhs);
    return Element(type, complexResult);
  }

  report_fatal_error(invalidArgument("Unsupported element type: %s",
                                     debugString(type).c_str()));
}

template <typename FloatFn, typename ComplexFn>
Element mapWithUpcastToDouble(const Element &el, FloatFn floatFn,
                              ComplexFn complexFn) {
  Type type = el.getType();

  if (isSupportedFloatType(type)) {
    APFloat elVal = el.getFloatValue();
    const llvm::fltSemantics &elSemantics = elVal.getSemantics();
    APFloat resultVal(floatFn(elVal.convertToDouble()));
    bool roundingErr;
    resultVal.convert(elSemantics, APFloat::rmNearestTiesToEven, &roundingErr);
    return Element(type, resultVal);
  }

  if (isSupportedComplexType(type)) {
    auto elVal = el.getComplexValue();
    const llvm::fltSemantics &elSemantics = elVal.real().getSemantics();
    auto resultVal = complexFn(std::complex<double>(
        elVal.real().convertToDouble(), elVal.imag().convertToDouble()));
    bool roundingErr;
    APFloat resultReal(resultVal.real());
    resultReal.convert(elSemantics, APFloat::rmNearestTiesToEven, &roundingErr);
    APFloat resultImag(resultVal.imag());
    resultImag.convert(elSemantics, APFloat::rmNearestTiesToEven, &roundingErr);
    return Element(type, std::complex<APFloat>(resultReal, resultImag));
  }

  report_fatal_error(invalidArgument("Unsupported element type: %s",
                                     debugString(type).c_str()));
}

}  // namespace

APInt Element::getIntegerValue() const {
  if (!isSupportedIntegerType(type_))
    report_fatal_error(StringRef("Accessing value of a type different than "
                                 "what is stored in Element object") +
                       LLVM_PRETTY_FUNCTION);

  return std::get<APInt>(value_);
}

APFloat Element::getFloatValue() const {
  if (!isSupportedFloatType(type_))
    report_fatal_error(StringRef("Accessing value of a type different than "
                                 "what is stored in Element object") +
                       LLVM_PRETTY_FUNCTION);

  return std::get<APFloat>(value_);
}

std::complex<APFloat> Element::getComplexValue() const {
  if (!isSupportedComplexType(type_))
    report_fatal_error(StringRef("Accessing value of a type different than "
                                 "what is stored in Element object") +
                       LLVM_PRETTY_FUNCTION);

  auto floatPair = std::get<std::pair<APFloat, APFloat>>(value_);
  return std::complex<APFloat>(floatPair.first, floatPair.second);
}

Element Element::operator+(const Element &other) const {
  return map(
      *this, other, [](APInt lhs, APInt rhs) { return lhs + rhs; },
      [](APInt lhs, APInt rhs) { return lhs + rhs; },
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

Element Element::operator-() const {
  return map(
      *this, [&](APInt val) { return -val; }, [&](APFloat val) { return -val; },
      [](std::complex<APFloat> val) { return -val; });
}

Element Element::operator-(const Element &other) const {
  return map(
      *this, other, [](APInt lhs, APInt rhs) { return lhs - rhs; },
      [](APFloat lhs, APFloat rhs) { return lhs - rhs; },
      [](std::complex<APFloat> lhs, std::complex<APFloat> rhs) {
        // NOTE: lhs - rhs doesn't work for std::complex<APFloat>
        // because the default implementation for the std::complex template
        // needs operator-= which is not defined on APFloat.
        auto resultReal = lhs.real() - rhs.real();
        auto resultImag = lhs.imag() - rhs.imag();
        return std::complex<APFloat>(resultReal, resultImag);
      });
}

Element ceil(const Element &el) {
  APFloat val = el.getFloatValue();
  val.roundToIntegral(APFloat::rmTowardPositive);
  return Element(el.getType(), val);
}

Element floor(const Element &el) {
  APFloat val = el.getFloatValue();
  val.roundToIntegral(APFloat::rmTowardNegative);
  return Element(el.getType(), val);
}

Element cosine(const Element &el) {
  return mapWithUpcastToDouble(
      el, [](double e) { return std::cos(e); },
      [](std::complex<double> e) { return std::cos(e); });
}

Element max(const Element &e1, const Element &e2) {
  return map(
      e1, e2,
      [](APInt lhs, APInt rhs) { return llvm::APIntOps::smax(lhs, rhs); },
      [](APInt lhs, APInt rhs) { return llvm::APIntOps::umax(lhs, rhs); },
      [](APFloat lhs, APFloat rhs) { return llvm::maximum(lhs, rhs); },
      [](std::complex<APFloat> lhs, std::complex<APFloat> rhs) {
        auto isFloatGreaterThan = [](APFloat A,
                                     APFloat B) -> llvm::Optional<bool> {
          // APFloat::compare treats -0.0 == 0.0, but IEEE 754-2018 demands -0 <
          // +0.
          if (A.isZero() && B.isZero() && (A.isNegative() != B.isNegative()))
            return !A.isNegative();

          auto cmpRes = A.compare(B);
          if (cmpRes == APFloat::cmpGreaterThan) return true;
          if (cmpRes == APFloat::cmpLessThan) return false;
          if (cmpRes == APFloat::cmpUnordered) return A.isNaN();
          return llvm::NoneType();
        };

        // Lexicographic comparison starts with real parts.
        auto cmpReal = isFloatGreaterThan(lhs.real(), rhs.real());
        if (cmpReal) return *cmpReal ? lhs : rhs;

        // Real parts are equal, compare the imag parts.
        auto cmpImag = isFloatGreaterThan(lhs.imag(), rhs.imag());
        if (cmpImag) return *cmpImag ? lhs : rhs;

        // Both real and imag parts are equal, return any (say the left
        // operand).
        return lhs;
      });
}

Element min(const Element &e1, const Element &e2) {
  return map(
      e1, e2,
      [](APInt lhs, APInt rhs) { return llvm::APIntOps::smin(lhs, rhs); },
      [](APInt lhs, APInt rhs) { return llvm::APIntOps::umin(lhs, rhs); },
      [](APFloat lhs, APFloat rhs) { return llvm::minimum(lhs, rhs); },
      [](std::complex<APFloat> lhs, std::complex<APFloat> rhs) {
        auto isFloatLessThan = [](APFloat A,
                                  APFloat B) -> llvm::Optional<bool> {
          // APFloat::compare treats -0.0 == 0.0, but IEEE 754-2018 demands -0 <
          // +0.
          if (A.isZero() && B.isZero() && (A.isNegative() != B.isNegative()))
            return A.isNegative();

          auto cmpRes = A.compare(B);
          if (cmpRes == APFloat::cmpLessThan) return true;
          if (cmpRes == APFloat::cmpGreaterThan) return false;
          if (cmpRes == APFloat::cmpUnordered) return A.isNaN();
          return llvm::NoneType();
        };

        // Lexicographic comparison starts with real parts.
        auto cmpReal = isFloatLessThan(lhs.real(), rhs.real());
        if (cmpReal) return *cmpReal ? lhs : rhs;

        // Real parts are equal, compare the imag parts.
        auto cmpImag = isFloatLessThan(lhs.imag(), rhs.imag());
        if (cmpImag) return *cmpImag ? lhs : rhs;

        // Both real and imag parts are equal, return any (say the left
        // operand).
        return lhs;
      });
}

Element sine(const Element &el) {
  return mapWithUpcastToDouble(
      el, [](double e) { return std::sin(e); },
      [](std::complex<double> e) { return std::sin(e); });
}

Element tanh(const Element &el) {
  return mapWithUpcastToDouble(
      el, [](double e) { return std::tanh(e); },
      [](std::complex<double> e) { return std::tanh(e); });
}

void Element::print(raw_ostream &os) const {
  if (isSupportedIntegerType(type_)) {
    IntegerAttr::get(type_, getIntegerValue()).print(os);
    return;
  }

  if (isSupportedFloatType(type_)) {
    FloatAttr::get(type_, getFloatValue()).print(os);
    return;
  }

  if (isSupportedComplexType(type_)) {
    auto complexElemTy = type_.dyn_cast<mlir::ComplexType>().getElementType();
    auto complexVal = getComplexValue();

    os << "[";
    FloatAttr::get(complexElemTy, complexVal.real()).print(os);
    os << ", ";
    FloatAttr::get(complexElemTy, complexVal.imag()).print(os);
    os << "]";

    return;
  }
}

void Element::dump() const { print(llvm::errs()); }

}  // namespace stablehlo
}  // namespace mlir
