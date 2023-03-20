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
#include "llvm/Support/Error.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Support/DebugStringHelper.h"
#include "stablehlo/reference/Errors.h"
#include "stablehlo/reference/Types.h"

namespace mlir {
namespace stablehlo {

namespace {

template <typename IntegerFn, typename BooleanFn, typename FloatFn,
          typename ComplexFn>
Element map(const Element &el, IntegerFn integerFn, BooleanFn boolFn,
            FloatFn floatFn, ComplexFn complexFn) {
  Type type = el.getType();

  if (isSupportedIntegerType(type)) {
    auto intEl = el.getIntegerValue();
    return Element(type, integerFn(intEl));
  }

  if (isSupportedBooleanType(type)) {
    auto boolEl = el.getBooleanValue();
    return Element(type, boolFn(boolEl));
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

template <typename IntegerFn, typename BooleanFn, typename FloatFn,
          typename ComplexFn>
Element map(const Element &lhs, const Element &rhs, IntegerFn integerFn,
            BooleanFn boolFn, FloatFn floatFn, ComplexFn complexFn) {
  Type type = lhs.getType();
  if (lhs.getType() != rhs.getType())
    report_fatal_error(invalidArgument("Element types don't match: %s vs %s",
                                       debugString(lhs.getType()).c_str(),
                                       debugString(rhs.getType()).c_str()));

  if (isSupportedIntegerType(type)) {
    auto intLhs = lhs.getIntegerValue();
    auto intRhs = rhs.getIntegerValue();
    return Element(type, integerFn(intLhs, intRhs));
  }

  if (isSupportedBooleanType(type)) {
    auto boolLhs = lhs.getBooleanValue();
    auto boolRhs = rhs.getBooleanValue();
    return Element(type, boolFn(boolLhs, boolRhs));
  }

  if (isSupportedFloatType(type)) {
    auto floatLhs = lhs.getFloatValue();
    auto floatRhs = rhs.getFloatValue();
    return Element(type, floatFn(floatLhs, floatRhs));
  }

  if (isSupportedComplexType(type)) {
    auto complexLhs = lhs.getComplexValue();
    auto complexRhs = rhs.getComplexValue();
    return Element(type, complexFn(complexLhs, complexRhs));
  }

  report_fatal_error(invalidArgument("Unsupported element type: %s",
                                     debugString(type).c_str()));
}

template <typename FloatFn, typename ComplexFn>
Element mapWithUpcastToDouble(const Element &el, FloatFn floatFn,
                              ComplexFn complexFn) {
  Type type = el.getType();

  if (isSupportedFloatType(type))
    return Element(type, floatFn(el.getFloatValue().convertToDouble()));

  if (isSupportedComplexType(type))
    return Element(type, complexFn(std::complex<double>(
                             el.getComplexValue().real().convertToDouble(),
                             el.getComplexValue().imag().convertToDouble())));

  report_fatal_error(invalidArgument("Unsupported element type: %s",
                                     debugString(type).c_str()));
}

template <typename FloatFn, typename ComplexFn>
Element mapWithUpcastToDouble(const Element &lhs, const Element &rhs,
                              FloatFn floatFn, ComplexFn complexFn) {
  Type type = lhs.getType();
  if (lhs.getType() != rhs.getType())
    report_fatal_error(invalidArgument("Element types don't match: %s vs %s",
                                       debugString(lhs.getType()).c_str(),
                                       debugString(rhs.getType()).c_str()));

  if (isSupportedFloatType(type)) {
    return Element(type, floatFn(lhs.getFloatValue().convertToDouble(),
                                 rhs.getFloatValue().convertToDouble()));
  }

  if (isSupportedComplexType(type)) {
    return Element(
        type, complexFn(std::complex<double>(
                            lhs.getComplexValue().real().convertToDouble(),
                            lhs.getComplexValue().imag().convertToDouble()),
                        std::complex<double>(
                            rhs.getComplexValue().real().convertToDouble(),
                            rhs.getComplexValue().imag().convertToDouble())));
  }

  report_fatal_error(invalidArgument("Unsupported element type: %s",
                                     debugString(type).c_str()));
}

// Checks if two APFloat values, f and g, are almost equal.
bool areApproximatelyEqual(APFloat f, APFloat g) {
  if (&f.getSemantics() != &g.getSemantics()) return false;

  llvm::APFloatBase::cmpResult cmpResult = f.compare(g);
  if (cmpResult == APFloat::cmpEqual) return true;
  if (cmpResult == APFloat::cmpUnordered) return f.isNaN() == g.isNaN();
  if (!f.isFiniteNonZero() || !g.isFiniteNonZero()) {
    // f and g could have the following cases
    //          f                           g
    //  (1) non-finite or zero      non-finite or zero
    //  (2) non-finite or zero      finite and non-zero
    //    (2.1) non-finite              finite and non-zero
    //    (2.2) zero                    finite and non-zero
    //  (3) finite and non-zero     non-finite or zero
    //    (3.1) finite and non-zero     non-finite
    //    (3.2) finite and non-zero     zero
    //
    // The equality of the cases (1), (2.1) and (3.1) have already been handled,
    // and it safe to return false for these case. The remaining cases (2.2)
    // and (3.2) will be handled in the following code.
    if (!f.isZero() && !g.isZero()) return false;
  }

  // Both f and g are finite (zero, subnormal, or normal) values.
  if (f.isNegative() != g.isNegative()) return false;
  return std::fabs(f.convertToDouble() - g.convertToDouble()) <= 0.0001;
}

}  // namespace

Element::Element(Type type, APInt value) : type_(type), value_(value) {}

Element::Element(Type type, int64_t value) {
  if (!isSupportedIntegerType(type))
    report_fatal_error(invalidArgument("Unsupported element type: %s",
                                       debugString(type).c_str()));
  type_ = type;
  value_ = APInt(type.getIntOrFloatBitWidth(), value,
                 /*isSigned=*/isSupportedSignedIntegerType(type));
}

Element::Element(Type type, bool value) : type_(type), value_(value) {}

Element::Element(Type type, APFloat value) : type_(type), value_(value) {}

Element::Element(Type type, double value) {
  if (isSupportedFloatType(type)) {
    APFloat floatVal(value);
    bool roundingErr;
    floatVal.convert(type.cast<FloatType>().getFloatSemantics(),
                     APFloat::rmNearestTiesToEven, &roundingErr);
    type_ = type;
    value_ = floatVal;
  } else if (isSupportedComplexType(type)) {
    APFloat real(value);
    APFloat imag(0.0);
    auto floatTy = type.cast<ComplexType>().getElementType().cast<FloatType>();
    bool roundingErr;
    real.convert(floatTy.getFloatSemantics(), APFloat::rmNearestTiesToEven,
                 &roundingErr);
    imag.convert(floatTy.getFloatSemantics(), APFloat::rmNearestTiesToEven,
                 &roundingErr);
    type_ = type;
    value_ = std::make_pair(real, imag);
  } else {
    report_fatal_error(invalidArgument("Unsupported element type: %s",
                                       debugString(type).c_str()));
  }
}

Element::Element(Type type, std::complex<APFloat> value)
    : type_(type), value_(std::make_pair(value.real(), value.imag())) {}

Element::Element(Type type, std::complex<double> value) {
  if (!isSupportedComplexType(type))
    report_fatal_error(invalidArgument("Unsupported element type: %s",
                                       debugString(type).c_str()));
  APFloat real(value.real());
  APFloat imag(value.imag());
  auto floatTy = type.cast<ComplexType>().getElementType().cast<FloatType>();
  bool roundingErr;
  real.convert(floatTy.getFloatSemantics(), APFloat::rmNearestTiesToEven,
               &roundingErr);
  imag.convert(floatTy.getFloatSemantics(), APFloat::rmNearestTiesToEven,
               &roundingErr);
  type_ = type;
  value_ = std::make_pair(real, imag);
}

APInt Element::getIntegerValue() const {
  if (!isSupportedIntegerType(type_))
    llvm::report_fatal_error("Element is not an integer");

  return std::get<APInt>(value_);
}

bool Element::getBooleanValue() const {
  if (!isSupportedBooleanType(type_))
    llvm::report_fatal_error("Element is not a boolean");

  return std::get<bool>(value_);
}

APFloat Element::getFloatValue() const {
  if (!isSupportedFloatType(type_))
    llvm::report_fatal_error("Element is not a floating-point");

  return std::get<APFloat>(value_);
}

std::complex<APFloat> Element::getComplexValue() const {
  if (!isSupportedComplexType(type_))
    llvm::report_fatal_error("Element is not a complex value");

  auto floatPair = std::get<std::pair<APFloat, APFloat>>(value_);
  return std::complex<APFloat>(floatPair.first, floatPair.second);
}

bool Element::operator!=(const Element &other) const {
  return !(*this == other);
}

Element Element::operator&(const Element &other) const {
  return map(
      *this, other, [](APInt lhs, APInt rhs) { return lhs & rhs; },
      [](bool lhs, bool rhs) -> bool { return lhs & rhs; },
      [](APFloat lhs, APFloat rhs) -> APFloat {
        llvm::report_fatal_error("float & float is unsupported");
      },
      [](std::complex<APFloat> lhs,
         std::complex<APFloat> rhs) -> std::complex<APFloat> {
        llvm::report_fatal_error("complex & complex is unsupported");
      });
}

Element Element::operator*(const Element &other) const {
  return map(
      *this, other, [](APInt lhs, APInt rhs) { return lhs * rhs; },
      [](bool lhs, bool rhs) -> bool { return lhs & rhs; },
      [](APFloat lhs, APFloat rhs) { return lhs * rhs; },
      [](std::complex<APFloat> lhs, std::complex<APFloat> rhs) {
        // TODO(#226): Use std::complex::operator*
        auto resultReal = lhs.real() * rhs.real() - lhs.imag() * rhs.imag();
        auto resultImag = lhs.real() * rhs.imag() + lhs.imag() * rhs.real();
        return std::complex<APFloat>(resultReal, resultImag);
      });
}

Element Element::operator+(const Element &other) const {
  return map(
      *this, other, [](APInt lhs, APInt rhs) { return lhs + rhs; },
      [](bool lhs, bool rhs) -> bool { return lhs | rhs; },
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
      *this, [&](APInt val) { return -val; },
      [](bool val) -> bool {
        llvm::report_fatal_error("-bool is unsupported");
      },
      [&](APFloat val) { return -val; },
      [](std::complex<APFloat> val) { return -val; });
}

Element Element::operator-(const Element &other) const {
  return map(
      *this, other, [](APInt lhs, APInt rhs) { return lhs - rhs; },
      [](bool lhs, bool rhs) -> bool {
        llvm::report_fatal_error("bool - bool is unsupported");
      },
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

Element Element::operator/(const Element &other) const {
  auto lhs = *this;
  auto rhs = other;

  Type type = lhs.getType();
  if (lhs.getType() != rhs.getType())
    report_fatal_error(invalidArgument("Element types don't match: %s vs %s",
                                       debugString(lhs.getType()).c_str(),
                                       debugString(rhs.getType()).c_str()));

  if (isSupportedIntegerType(type)) {
    auto intLhs = lhs.getIntegerValue();
    auto intRhs = rhs.getIntegerValue();
    return Element(type, isSupportedSignedIntegerType(type)
                             ? intLhs.sdiv(intRhs)
                             : intLhs.udiv(intRhs));
  }

  if (isSupportedFloatType(type)) {
    APFloat lhsVal = lhs.getFloatValue();
    APFloat rhsVal = rhs.getFloatValue();
    return Element(type, lhsVal / rhsVal);
  }

  if (isSupportedComplexType(type)) {
    auto lhsVal = lhs.getComplexValue();
    auto rhsVal = rhs.getComplexValue();
    const llvm::fltSemantics &elSemantics = lhsVal.real().getSemantics();
    auto resultVal = std::complex<double>(lhsVal.real().convertToDouble(),
                                          lhsVal.imag().convertToDouble()) /
                     std::complex<double>(rhsVal.real().convertToDouble(),
                                          rhsVal.imag().convertToDouble());
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

bool Element::operator<(const Element &other) const {
  Type type = other.getType();
  if (type_ != type)
    report_fatal_error(invalidArgument("Element types don't match: %s vs %s",
                                       debugString(type_).c_str(),
                                       debugString(type).c_str()));

  if (isSupportedIntegerType(type)) {
    auto intLhs = getIntegerValue();
    auto intRhs = other.getIntegerValue();
    return isSupportedSignedIntegerType(type) ? intLhs.slt(intRhs)
                                              : intLhs.ult(intRhs);
  }

  if (isSupportedBooleanType(type)) {
    auto boolLhs = getBooleanValue();
    auto boolRhs = other.getBooleanValue();
    return boolLhs < boolRhs;
  }

  if (isSupportedFloatType(type)) {
    auto floatLhs = getFloatValue();
    auto floatRhs = other.getFloatValue();
    return floatLhs < floatRhs;
  }

  report_fatal_error(invalidArgument("Unsupported element type: %s",
                                     debugString(type).c_str()));
}

bool Element::operator<=(const Element &other) const {
  return (*this < other) || (*this == other);
}

bool Element::operator==(const Element &other) const {
  Type type = other.getType();
  if (type_ != type)
    report_fatal_error(invalidArgument("Element types don't match: %s vs %s",
                                       debugString(type_).c_str(),
                                       debugString(type).c_str()));

  if (isSupportedIntegerType(type)) {
    auto intLhs = getIntegerValue();
    auto intRhs = other.getIntegerValue();
    return intLhs == intRhs;
  }

  if (isSupportedBooleanType(type)) {
    auto boolLhs = getBooleanValue();
    auto boolRhs = other.getBooleanValue();
    return boolLhs == boolRhs;
  }

  if (isSupportedFloatType(type)) {
    auto floatLhs = getFloatValue();
    auto floatRhs = other.getFloatValue();
    return floatLhs == floatRhs;
  }

  if (isSupportedComplexType(type)) {
    auto complexLhs = getComplexValue();
    auto complexRhs = other.getComplexValue();
    return complexLhs.real() == complexRhs.real() &&
           complexLhs.imag() == complexRhs.imag();
  }

  report_fatal_error(invalidArgument("Unsupported element type: %s",
                                     debugString(type).c_str()));
}

bool Element::operator>(const Element &other) const {
  Type type = other.getType();
  if (type_ != type)
    report_fatal_error(invalidArgument("Element types don't match: %s vs %s",
                                       debugString(type_).c_str(),
                                       debugString(type).c_str()));

  if (isSupportedIntegerType(type)) {
    auto intLhs = getIntegerValue();
    auto intRhs = other.getIntegerValue();
    return isSupportedSignedIntegerType(type) ? intLhs.sgt(intRhs)
                                              : intLhs.ugt(intRhs);
  }

  if (isSupportedBooleanType(type)) {
    auto boolLhs = getBooleanValue();
    auto boolRhs = other.getBooleanValue();
    return boolLhs > boolRhs;
  }

  if (isSupportedFloatType(type)) {
    auto floatLhs = getFloatValue();
    auto floatRhs = other.getFloatValue();
    return floatLhs > floatRhs;
  }

  report_fatal_error(invalidArgument("Unsupported element type: %s",
                                     debugString(type).c_str()));
}

bool Element::operator>=(const Element &other) const {
  return (*this > other) || (*this == other);
}

Element Element::operator^(const Element &other) const {
  return map(
      *this, other, [](APInt lhs, APInt rhs) { return lhs ^ rhs; },
      [](bool lhs, bool rhs) -> bool { return lhs ^ rhs; },
      [](APFloat lhs, APFloat rhs) -> APFloat {
        llvm::report_fatal_error("float ^ float is unsupported");
      },
      [](std::complex<APFloat> lhs,
         std::complex<APFloat> rhs) -> std::complex<APFloat> {
        llvm::report_fatal_error("complex ^ complex is unsupported");
      });
}

Element Element::operator|(const Element &other) const {
  return map(
      *this, other, [](APInt lhs, APInt rhs) { return lhs | rhs; },
      [](bool lhs, bool rhs) -> bool { return lhs | rhs; },
      [](APFloat lhs, APFloat rhs) -> APFloat {
        llvm::report_fatal_error("float | float is unsupported");
      },
      [](std::complex<APFloat> lhs,
         std::complex<APFloat> rhs) -> std::complex<APFloat> {
        llvm::report_fatal_error("complex | complex is unsupported");
      });
}

Element Element::operator~() const {
  return map(
      *this, [](APInt val) { return ~val; },
      [](bool val) -> bool { return !val; },
      [](APFloat val) -> APFloat {
        llvm::report_fatal_error("~float is unsupported");
      },
      [](std::complex<APFloat> val) -> std::complex<APFloat> {
        llvm::report_fatal_error("~complex is unsupported");
      });
}

Element abs(const Element &el) {
  Type type = el.getType();

  if (isSupportedIntegerType(el.getType())) {
    auto intEl = el.getIntegerValue();
    return Element(type, intEl.abs());
  }

  if (isSupportedFloatType(type)) {
    auto elVal = el.getFloatValue();
    return Element(type, llvm::abs(elVal));
  }

  if (isSupportedComplexType(type)) {
    auto elVal = el.getComplexValue();
    auto resultVal = std::abs(std::complex<double>(
        elVal.real().convertToDouble(), elVal.imag().convertToDouble()));
    return Element(type.cast<ComplexType>().getElementType(), resultVal);
  }

  report_fatal_error(invalidArgument("Unsupported element type: %s",
                                     debugString(type).c_str()));
}

bool areApproximatelyEqual(const Element &e1, const Element &e2) {
  Type type = e1.getType();
  if (type != e2.getType())
    report_fatal_error(invalidArgument("Element types don't match: %s vs %s",
                                       debugString(type).c_str(),
                                       debugString(e2.getType()).c_str()));

  if (isSupportedFloatType(type))
    return areApproximatelyEqual(e1.getFloatValue(), e2.getFloatValue());

  if (isSupportedComplexType(type)) {
    auto complexLhs = e1.getComplexValue();
    auto complexRhs = e2.getComplexValue();
    return areApproximatelyEqual(complexLhs.real(), complexRhs.real()) &&
           areApproximatelyEqual(complexLhs.imag(), complexRhs.imag());
  }

  report_fatal_error(invalidArgument("Unsupported element type: %s",
                                     debugString(type).c_str()));
}

Element ceil(const Element &el) {
  APFloat val = el.getFloatValue();
  val.roundToIntegral(APFloat::rmTowardPositive);
  return Element(el.getType(), val);
}

Element exponential(const Element &el) {
  return mapWithUpcastToDouble(
      el, [](double e) { return std::exp(e); },
      [](std::complex<double> e) { return std::exp(e); });
}

Element floor(const Element &el) {
  APFloat val = el.getFloatValue();
  val.roundToIntegral(APFloat::rmTowardNegative);
  return Element(el.getType(), val);
}

Element imag(const Element &el) {
  if (isSupportedFloatType(el.getType())) {
    const llvm::fltSemantics &elSemantics = el.getFloatValue().getSemantics();
    bool roundingErr;
    APFloat resultImag(0.0);
    resultImag.convert(elSemantics, APFloat::rmNearestTiesToEven, &roundingErr);
    return Element(el.getType(), resultImag);
  }
  if (isSupportedComplexType(el.getType()))
    return Element(el.getType().cast<ComplexType>().getElementType(),
                   el.getComplexValue().imag());
  report_fatal_error(invalidArgument("Unsupported element type: %s",
                                     debugString(el.getType()).c_str()));
}

Element cosine(const Element &el) {
  return mapWithUpcastToDouble(
      el, [](double e) { return std::cos(e); },
      [](std::complex<double> e) { return std::cos(e); });
}

Element log(const Element &el) {
  return mapWithUpcastToDouble(
      el, [](double e) { return std::log(e); },
      [](std::complex<double> e) { return std::log(e); });
}

Element logistic(const Element &el) {
  auto one = Element(el.getType(), 1.0);
  return one / (one + exponential(-el));
}

Element max(const Element &e1, const Element &e2) {
  return map(
      e1, e2,
      [&](APInt lhs, APInt rhs) {
        return isSupportedSignedIntegerType(e1.getType())
                   ? llvm::APIntOps::smax(lhs, rhs)
                   : llvm::APIntOps::umax(lhs, rhs);
      },
      [](bool lhs, bool rhs) -> bool { return lhs | rhs; },
      [](APFloat lhs, APFloat rhs) { return llvm::maximum(lhs, rhs); },
      [](std::complex<APFloat> lhs, std::complex<APFloat> rhs) {
        auto cmpRes = lhs.real().compare(rhs.real()) == APFloat::cmpEqual
                          ? lhs.imag() > rhs.imag()
                          : lhs.real() > rhs.real();
        return cmpRes ? lhs : rhs;
      });
}

Element min(const Element &e1, const Element &e2) {
  return map(
      e1, e2,
      [&](APInt lhs, APInt rhs) {
        return isSupportedSignedIntegerType(e1.getType())
                   ? llvm::APIntOps::smin(lhs, rhs)
                   : llvm::APIntOps::umin(lhs, rhs);
      },
      [](bool lhs, bool rhs) -> bool { return lhs & rhs; },
      [](APFloat lhs, APFloat rhs) { return llvm::minimum(lhs, rhs); },
      [](std::complex<APFloat> lhs, std::complex<APFloat> rhs) {
        auto cmpRes = lhs.real().compare(rhs.real()) == APFloat::cmpEqual
                          ? lhs.imag() < rhs.imag()
                          : lhs.real() < rhs.real();
        return cmpRes ? lhs : rhs;
      });
}

Element power(const Element &e1, const Element &e2) {
  Type type = e1.getType();

  if (isSupportedIntegerType(type)) {
    bool isSigned = isSupportedSignedIntegerType(type);
    APInt base = e1.getIntegerValue();
    APInt exponent = e2.getIntegerValue();
    if (isSigned && exponent.isNegative()) {
      if (base.abs().isOne())
        exponent = exponent.abs();
      else
        return Element(type, static_cast<int64_t>(0));
    }
    APInt result(base.getBitWidth(), 1, isSigned);
    while (!exponent.isZero()) {
      if (!(exponent & 1).isZero()) result *= base;
      base *= base;
      exponent = exponent.lshr(1);
    }
    return Element(type, result);
  }

  return mapWithUpcastToDouble(
      e1, e2, [](double lhs, double rhs) { return std::pow(lhs, rhs); },
      [](std::complex<double> lhs, std::complex<double> rhs) {
        return std::pow(lhs, rhs);
      });
}

Element real(const Element &el) {
  if (isSupportedFloatType(el.getType())) return el;
  if (isSupportedComplexType(el.getType()))
    return Element(el.getType().cast<ComplexType>().getElementType(),
                   el.getComplexValue().real());
  report_fatal_error(invalidArgument("Unsupported element type: %s",
                                     debugString(el.getType()).c_str()));
}

Element rem(const Element &e1, const Element &e2) {
  return map(
      e1, e2,
      [&](APInt lhs, APInt rhs) {
        return isSupportedSignedIntegerType(e1.getType()) ? lhs.srem(rhs)
                                                          : lhs.urem(rhs);
      },
      [](bool lhs, bool rhs) -> bool {
        llvm::report_fatal_error("rem(bool, bool) is unsupported");
      },
      [](APFloat lhs, APFloat rhs) {
        // APFloat::fmod VS APFloat:remainder: the returned value of the latter
        // is not guaranteed to have the same sign as lhs. So mod() is preferred
        // here. The returned "APFloat::opStatus" is ignored.
        (void)lhs.mod(rhs);
        return lhs;
      },
      [](std::complex<APFloat> lhs,
         std::complex<APFloat> rhs) -> std::complex<APFloat> {
        // TODO(#997): remove support for complex
        llvm::report_fatal_error("rem(complex, complex) is not implemented");
      });
}

Element rsqrt(const Element &el) {
  return mapWithUpcastToDouble(
      el, [](double e) { return 1.0 / std::sqrt(e); },
      [](std::complex<double> e) { return 1.0 / std::sqrt(e); });
}

Element sine(const Element &el) {
  return mapWithUpcastToDouble(
      el, [](double e) { return std::sin(e); },
      [](std::complex<double> e) { return std::sin(e); });
}

Element sqrt(const Element &el) {
  return mapWithUpcastToDouble(
      el, [](double e) { return std::sqrt(e); },
      [](std::complex<double> e) { return std::sqrt(e); });
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

  if (isSupportedBooleanType(type_)) {
    IntegerAttr::get(type_, getBooleanValue()).print(os);
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
