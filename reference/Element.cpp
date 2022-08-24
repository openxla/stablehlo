#include "Element.h"

#include <complex>

#include "llvm/ADT/APFloat.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir {
namespace stablehlo {

namespace {

bool areApproximatelyEqual(APFloat f, APFloat g) {
  llvm::APFloatBase::cmpResult cmpResult = f.compare(g);
  if (cmpResult == APFloat::cmpEqual) return true;

  if (cmpResult == APFloat::cmpUnordered) return f.isNaN() == g.isNaN();

  auto absFloat = abs(f - g);
  APFloat err(f.getSemantics(), "1e-5");
  return absFloat.compare(err) == APFloat::cmpLessThan;
}

bool isSupportedUnsignedIntegerType(Type type) {
  return type.isUnsignedInteger(4) || type.isUnsignedInteger(8) ||
         type.isUnsignedInteger(16) || type.isUnsignedInteger(32) ||
         type.isUnsignedInteger(64);
}

bool isSupportedSignedIntegerType(Type type) {
  return type.isSignedInteger(4) || type.isSignedInteger(8) ||
         type.isSignedInteger(16) || type.isSignedInteger(32) ||
         type.isSignedInteger(64);
}

bool isSupportedIntegerType(Type type) {
  return type.isSignlessInteger(1) || isSupportedUnsignedIntegerType(type) ||
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

  llvm_unreachable("Unsupported element type");
}

bool Element::operator==(const Element &other) const {
  auto type = getType();
  assert(type == other.getType());

  if (isSupportedFloatType(type)) {
    auto left = getFloatValue(*this);
    auto right = getFloatValue(other);
    return areApproximatelyEqual(left, right);
  }

  if (isSupportedIntegerType(type)) {
    auto left = getIntegerValue(*this);
    auto right = getIntegerValue(other);
    return left == right;
  }

  if (isSupportedComplexType(type)) {
    auto leftComplexValue = getComplexValue(*this);
    auto rightComplexValue = getComplexValue(other);

    return areApproximatelyEqual(leftComplexValue.real(),
                                 rightComplexValue.real()) &&
           areApproximatelyEqual(leftComplexValue.imag(),
                                 rightComplexValue.imag());
  }

  llvm_unreachable("Unsupported element type");
}

bool Element::operator!=(const Element &other) const {
  return !(*this == other);
}

void Element::print(raw_ostream &os) const { value_.print(os); }

void Element::dump() const {
  print(llvm::errs());
  llvm::errs() << "\n";
}

}  // namespace stablehlo
}  // namespace mlir
