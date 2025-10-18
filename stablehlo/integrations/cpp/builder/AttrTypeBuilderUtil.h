/* Copyright 2025 The OpenXLA Authors.

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

#ifndef STABLEHLO_BUILDER_ATTRTYPEBUILDERUTIL_H_
#define STABLEHLO_BUILDER_ATTRTYPEBUILDERUTIL_H_

#include <complex>
#include <cstdint>
#include <type_traits>
#include <vector>

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"

namespace mlir {

//////////////////////
// Builders - Location
//////////////////////

Location unknownLoc(MLIRContext& ctx);
Location fileLineColLoc(MLIRContext& ctx, StringRef file, int64_t line,
                        int64_t col);

//////////////////////
// Builders - Tensor Types
//////////////////////

// POD type to Tensor element type map

// List of supported Tensor Element Types.
// This list is fairly XLA specific, used to provide sugar for the common
// RankedTensorType's we'll need to build.
enum class ElementType {
  // clang-format off
  PRED,
  I2, I4, I8, I16, I32, I64,
  UI2, UI4, UI8, UI16, UI32, UI64,
  BF16, F16, F32, F64,
  F4E2M1FN, F6E2M3FN, F6E3M2FN, F8E3M4, F8E4M3,
  F8E4M3FN, F8E4M3FNUZ, F8E4M3B11FNUZ, F8E5M2, F8E5M2FNUZ, F8E8M0FNU,
  COMPLEXF32, COMPLEXF64
  // clang-format on
};

bool IsBoolean(ElementType elementType);

bool IsComplex(ElementType elementType);

bool IsFloat(ElementType elementType);

bool IsInteger(ElementType elementType, bool includeBool);

// In StableHLO, we refer to signed integer as the MLIR's equivalent signless
// integer. StableHLO does not have a notion of signless integers like MLIR.
bool IsSignedInteger(ElementType elementType);

bool IsUnsignedInteger(ElementType elementType);

Type getElementType(MLIRContext& ctx, ElementType elementType);

// Build a ranked tensor type with an element type of ElementType.
RankedTensorType makeTensorType(MLIRContext& ctx, ArrayRef<int64_t> shape,
                                ElementType elementType);

// Build a ranked tensor type with an MLIR element type.
RankedTensorType makeTensorType(MLIRContext& ctx, ArrayRef<int64_t> shape,
                                Type elementType);

//////////////////////
// Builders - Constant Literals
//////////////////////

namespace detail {

APFloat toAPFloat(double val, FloatType floatType);

//////////////////////
// Literal Conversion - Int
//////////////////////
inline IntegerAttr getIntegerAttr(llvm::APSInt value, IntegerType type) {
  value.setIsSigned(type.isSigned());
  APSInt ext = value.extOrTrunc(type.getIntOrFloatBitWidth());
  return IntegerAttr::get(type, ext);
}
template <typename T>
typename std::enable_if<std::is_integral<T>::value, IntegerAttr>::type
getIntegerAttr(T value, IntegerType type) {
  return getIntegerAttr(APSInt::get(value), type);
}
inline IntegerAttr getIntegerAttr(double value, IntegerType type) {
  return getIntegerAttr(static_cast<int64_t>(value), type);
}
inline IntegerAttr getIntegerAttr(llvm::APFloat value, IntegerType type) {
  return getIntegerAttr(value.convertToDouble(), type);
}
template <typename T>
inline IntegerAttr getIntegerAttr(std::complex<T> value, IntegerType type) {
  return getIntegerAttr(value.real(), type);
}

template <typename T>
SmallVector<Attribute> getIntegerAttrs(ArrayRef<T> values, IntegerType type) {
  return llvm::to_vector(llvm::map_range(values, [&](T value) -> Attribute {
    return getIntegerAttr(value, type);
  }));
}

//////////////////////
// Literal Conversion - Float
//////////////////////
inline FloatAttr getFloatAttr(llvm::APFloat value, FloatType type) {
  return FloatAttr::get(type, value);
}
inline FloatAttr getFloatAttr(double value, FloatType type) {
  return getFloatAttr(toAPFloat(value, type), type);
}
template <typename T>
typename std::enable_if<std::is_integral<T>::value, FloatAttr>::type
getFloatAttr(T value, FloatType type) {
  return getFloatAttr(static_cast<double>(value), type);
}
inline FloatAttr getFloatAttr(llvm::APSInt value, FloatType type) {
  return getFloatAttr(value.roundToDouble(), type);
}
template <typename T>
FloatAttr getFloatAttr(std::complex<T> value, FloatType type) {
  return getFloatAttr(value.real(), type);
}

template <typename T>
SmallVector<Attribute> getFloatAttrs(ArrayRef<T> values, FloatType type) {
  return llvm::to_vector(llvm::map_range(
      values, [&](T value) -> Attribute { return getFloatAttr(value, type); }));
}

//////////////////////
// Literal Conversion - Complex
//////////////////////
template <typename T>
typename std::enable_if<std::is_floating_point_v<T>,
                        std::complex<APFloat>>::type
getComplexValue(std::complex<T> value, FloatType floatType) {
  FloatAttr realAttr = getFloatAttr(value.real(), floatType);
  FloatAttr imagAttr = getFloatAttr(value.imag(), floatType);
  return std::complex<APFloat>(realAttr.getValue(), imagAttr.getValue());
}
template <typename T>
std::complex<APFloat> getComplexValue(T value, FloatType floatType) {
  FloatAttr realAttr = getFloatAttr(value, floatType);
  return std::complex<APFloat>(realAttr.getValue(), toAPFloat(0.0, floatType));
}
template <typename T>
SmallVector<std::complex<APFloat>> getComplexValues(ArrayRef<T> values,
                                                    FloatType floatType) {
  return llvm::to_vector(
      llvm::map_range(values, [&](T value) -> std::complex<APFloat> {
        return getComplexValue(value, floatType);
      }));
}

}  // namespace detail

// Creates a DenseElementsAttr from a single value (splat) and a target
// RankedTensorType.
//
// This function attempts to create a DenseElementsAttr by broadcasting the
// provided `value` to the shape specified in `tensorType`. It supports
// IntegerType, FloatType, and ComplexType.
//
// Supported input types (`T`):
//   - For IntegerType: any arithmetic type. Arithmetic types will
//   be cast to `int64_t`.
//   - For FloatType: any arithmetic type. Arithmetic types will
//   be cast to `double`.
//   - For ComplexType: `std::complex<T>` and any arithmetic type. The
//     imaginary part will be set to 0 if the input is an arithmetic type.
//
// Args:
//   value: The value to broadcast.
//   tensorType: The target RankedTensorType.
//
// Returns:
//   A splat DenseElementsAttr.
//
// Raises fatal exception if the element type is unsupported.
template <typename T>
DenseElementsAttr makeConstant(T value, RankedTensorType tensorType) {
  return TypeSwitch<Type, DenseElementsAttr>(tensorType.getElementType())
      .template Case<IntegerType>([&](IntegerType type) -> DenseElementsAttr {
        IntegerAttr intAttr = detail::getIntegerAttr(value, type);
        return DenseElementsAttr::get(tensorType, intAttr);
      })
      .template Case<FloatType>([&](FloatType type) -> DenseElementsAttr {
        FloatAttr floatAttr = detail::getFloatAttr(value, type);
        return DenseElementsAttr::get(tensorType, floatAttr);
      })
      .template Case<ComplexType>([&](ComplexType type) -> DenseElementsAttr {
        auto floatType = dyn_cast<FloatType>(type.getElementType());
        if (!floatType)
          llvm::report_fatal_error(
              "makeConstant with non-float complex type is unsupported.");

        std::complex<APFloat> complexValue =
            detail::getComplexValue(value, floatType);
        return DenseElementsAttr::get(tensorType, complexValue);
      })
      .Default([](Type) -> DenseElementsAttr {
        llvm::report_fatal_error(
            "makeConstant called with unsupported MLIR type, must be "
            "IntegerType, FloatType, or ComplexType.");
        return nullptr;
      });
}

// Creates a DenseElementsAttr from a list of values and a target
// RankedTensorType.
//
// This function may perform some literal coercions if the tensor element type
// does not match the provided value type.
//
// See `makeConstant(T value, RankedTensorType tensorType)` for full details.
template <typename T>
DenseElementsAttr makeConstant(ArrayRef<T> values,
                               RankedTensorType tensorType) {
  return TypeSwitch<Type, DenseElementsAttr>(tensorType.getElementType())
      .template Case<IntegerType>([&](IntegerType type) -> DenseElementsAttr {
        SmallVector<Attribute> intAttrs = detail::getIntegerAttrs(values, type);
        return DenseElementsAttr::get(tensorType, intAttrs);
      })
      .template Case<FloatType>([&](FloatType type) -> DenseElementsAttr {
        SmallVector<Attribute> floatAttrs = detail::getFloatAttrs(values, type);
        return DenseElementsAttr::get(tensorType, floatAttrs);
      })
      .template Case<ComplexType>([&](ComplexType type) -> DenseElementsAttr {
        auto floatType = dyn_cast<FloatType>(type.getElementType());
        if (!floatType)
          llvm::report_fatal_error(
              "makeConstant with non-float complex type is unsupported.");

        SmallVector<std::complex<APFloat>> complexValues =
            detail::getComplexValues(values, floatType);
        return DenseElementsAttr::get(tensorType, complexValues);
      })
      .Default([&](Type) -> DenseElementsAttr {
        llvm::report_fatal_error(
            "makeConstant called with unsupported MLIR type, must be "
            "IntegerType, FloatType, or ComplexType.");
        return nullptr;
      });
}

template <typename T>
DenseElementsAttr makeConstant(const SmallVector<T>& values,
                               RankedTensorType tensorType) {
  // Force the compiler to call the overload that takes an SmallVector<T> to
  // prevent infinite loops
  return makeConstant(ArrayRef<T>(values), tensorType);
}

template <typename T>
DenseElementsAttr makeConstant(const std::vector<T>& values,
                               RankedTensorType tensorType) {
  // Force the compiler to call the overload that takes an ArrayRef<T> to
  // prevent infinite loops.
  return makeConstant(ArrayRef<T>(values), tensorType);
}

}  // namespace mlir

#endif  // STABLEHLO_BUILDER_ATTRTYPEBUILDERUTIL_H_
