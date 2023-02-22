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

#ifndef STABLEHLO_REFERENCE_ELEMENT_H
#define STABLEHLO_REFERENCE_ELEMENT_H

#include <complex>
#include <variant>

#include "llvm/ADT/APFloat.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"

namespace mlir {
namespace stablehlo {

/// Class to represent an element of a tensor. An Element object stores the
/// element type of the tensor and, depending on that element type, a constant
/// value of type integer, floating-paint, or complex type.
class Element {
 public:
  /// \name Constructors
  /// @{
  Element(Type type, APInt value) : type_(type), value_(value) {}
  Element(Type type, bool value) : type_(type), value_(value) {}
  Element(Type type, APFloat value) : type_(type), value_(value) {}
  Element(Type type, std::complex<APFloat> value)
      : type_(type), value_(std::make_pair(value.real(), value.imag())) {}

  Element(const Element &other) = default;
  /// @}

  /// Assignment operator.
  Element &operator=(const Element &other) = default;

  /// Returns type of the Element object.
  Type getType() const { return type_; }

  /// Returns the underlying integer value stored in an Element object with
  /// integer type.
  APInt getIntegerValue() const;

  /// Returns the underlying boolean value stored in an Element object with
  /// bool type.
  bool getBooleanValue() const;

  /// Returns the underlying floating-point value stored in an Element object
  /// with floating-point type.
  APFloat getFloatValue() const;

  /// Returns the underlying complex value stored in an Element object with
  /// complex type.
  std::complex<APFloat> getComplexValue() const;

  /// Overloaded and (bitwise) operator.
  Element operator&(const Element &other) const;

  /// Overloaded add operator.
  Element operator+(const Element &other) const;

  /// Overloaded divide operator.
  Element operator/(const Element &other) const;

  /// Overloaded multiply operator.
  Element operator*(const Element &other) const;

  /// Overloaded negate operator.
  Element operator-() const;

  /// Overloaded subtract operator.
  Element operator-(const Element &other) const;

  /// Overloaded xor (bitwise) operator.
  Element operator^(const Element &other) const;

  /// Overloaded or (bitwise) operator.
  Element operator|(const Element &other) const;

  /// Overloaded not (bitwise) operator.
  Element operator~() const;

  /// Print utilities for Element objects.
  void print(raw_ostream &os) const;

  /// Print utilities for Element objects.
  void dump() const;

 private:
  Type type_;
  std::variant<APInt, bool, APFloat, std::pair<APFloat, APFloat>> value_;
};

/// Returns abs of Element object.
Element abs(const Element &e);

/// Returns ceil of Element object.
Element ceil(const Element &e);

/// Returns cosine of Element object.
Element cosine(const Element &e);

/// Returns exponential of Element object.
Element exponential(const Element &el);

/// Returns floor of Element object.
Element floor(const Element &e);

/// Returns log of Element object.
Element log(const Element &el);

/// Returns the maximum between two Element objects.
Element max(const Element &e1, const Element &e2);

/// Returns the minimum between two Element objects.
Element min(const Element &e1, const Element &e2);

/// Returns reverse square root of Element object.
Element rsqrt(const Element &e);

/// Returns sine of Element object.
Element sine(const Element &e);

/// Returns square root of Element object.
Element sqrt(const Element &e);

/// Returns tanh of Element object.
Element tanh(const Element &e);

/// Print utilities for Element objects.
inline raw_ostream &operator<<(raw_ostream &os, Element element) {
  element.print(os);
  return os;
}

}  // namespace stablehlo
}  // namespace mlir

#endif  // STABLEHLO_REFERENCE_ELEMENT_H
