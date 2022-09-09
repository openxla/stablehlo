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

#ifndef STABLHLO_REFERENCE_ELEMENT_H
#define STABLHLO_REFERENCE_ELEMENT_H

#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Types.h"

namespace mlir {
namespace stablehlo {

class Element {
 public:
  /// \name Constructors
  /// @{
  Element(Type type, Attribute value) : type_(type), value_(value) {}

  Element(const Element &other) = default;
  /// @}

  /// Assignment operator.
  Element &operator=(const Element &other) = default;

  /// Returns type of the Element object.
  Type getType() const { return type_; }

  /// Returns the underlying storage of Element object.
  Attribute getValue() const { return value_; }

  /// Overloaded + operator.
  Element operator+(const Element &other) const;

  /// Print utilities for Element objects.
  void print(raw_ostream &os) const;

  /// Print utilities for Element objects.
  void dump() const;

 private:
  Type type_;
  Attribute value_;
};

/// Print utilities for Element objects.
inline raw_ostream &operator<<(raw_ostream &os, Element element) {
  element.print(os);
  return os;
}

}  // namespace stablehlo
}  // namespace mlir

#endif  // STABLHLO_REFERENCE_ELEMENT_H
