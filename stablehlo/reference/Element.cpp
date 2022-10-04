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
