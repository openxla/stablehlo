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

  /// Overloaded == operator.
  bool operator==(const Element &other) const;

  /// Overloaded != operator.
  bool operator!=(const Element &other) const;

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
