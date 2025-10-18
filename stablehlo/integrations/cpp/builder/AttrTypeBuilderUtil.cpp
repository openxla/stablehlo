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

#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"

#include <cstdint>

#include "llvm/Support/ErrorHandling.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"

namespace mlir {

//////////////////////
// Builders - Location
//////////////////////

Location unknownLoc(MLIRContext& ctx) { return UnknownLoc::get(&ctx); }

Location fileLineColLoc(MLIRContext& ctx, StringRef file, int64_t line,
                        int64_t col) {
  return FileLineColLoc::get(&ctx, file, line, col);
}

//////////////////////
// Builders - Tensor Types
//////////////////////

Type getElementType(MLIRContext& ctx, ElementType elementType) {
  Builder builder(&ctx);

  switch (elementType) {
    case ElementType::PRED:
      return builder.getI1Type();
    case ElementType::I2:
      return builder.getI2Type();
    case ElementType::I4:
      return builder.getI4Type();
    case ElementType::I8:
      return builder.getI8Type();
    case ElementType::I16:
      return builder.getI16Type();
    case ElementType::I32:
      return builder.getI32Type();
    case ElementType::I64:
      return builder.getI64Type();
    case ElementType::UI2:
      return IntegerType::get(&ctx, 2, IntegerType::Unsigned);
    case ElementType::UI4:
      return IntegerType::get(&ctx, 4, IntegerType::Unsigned);
    case ElementType::UI8:
      return IntegerType::get(&ctx, 8, IntegerType::Unsigned);
    case ElementType::UI16:
      return IntegerType::get(&ctx, 16, IntegerType::Unsigned);
    case ElementType::UI32:
      return IntegerType::get(&ctx, 32, IntegerType::Unsigned);
    case ElementType::UI64:
      return IntegerType::get(&ctx, 64, IntegerType::Unsigned);
    case ElementType::BF16:
      return builder.getBF16Type();
    case ElementType::F16:
      return builder.getF16Type();
    case ElementType::F32:
      return builder.getF32Type();
    case ElementType::F64:
      return builder.getF64Type();
    case ElementType::F4E2M1FN:
      return Float4E2M1FNType::get(&ctx);
    case ElementType::F6E2M3FN:
      return Float6E2M3FNType::get(&ctx);
    case ElementType::F6E3M2FN:
      return Float6E3M2FNType::get(&ctx);
    case ElementType::F8E3M4:
      return Float8E3M4Type::get(&ctx);
    case ElementType::F8E4M3:
      return Float8E4M3Type::get(&ctx);
    case ElementType::F8E4M3FN:
      return Float8E4M3FNType::get(&ctx);
    case ElementType::F8E4M3FNUZ:
      return Float8E4M3FNUZType::get(&ctx);
    case ElementType::F8E4M3B11FNUZ:
      return Float8E4M3B11FNUZType::get(&ctx);
    case ElementType::F8E5M2:
      return Float8E5M2Type::get(&ctx);
    case ElementType::F8E5M2FNUZ:
      return Float8E5M2FNUZType::get(&ctx);
    case ElementType::F8E8M0FNU:
      return Float8E8M0FNUType::get(&ctx);
    case ElementType::COMPLEXF32:
      return ComplexType::get(builder.getF32Type());
    case ElementType::COMPLEXF64:
      return ComplexType::get(builder.getF64Type());
    default:
      llvm::report_fatal_error("Unsupported element type");
  }
}

bool IsBoolean(ElementType elementType) {
  MLIRContext ctx;
  return getElementType(ctx, elementType).isInteger(1);
}

bool IsComplex(ElementType elementType) {
  MLIRContext ctx;
  auto type = dyn_cast<ComplexType>(getElementType(ctx, elementType));
  return !!type;
}

bool IsFloat(ElementType elementType) {
  MLIRContext ctx;
  return getElementType(ctx, elementType).isFloat();
}

bool IsInteger(ElementType elementType, bool includeBool = false) {
  MLIRContext ctx;
  Type type = getElementType(ctx, elementType);
  return type.isInteger() && (includeBool || !IsBoolean(elementType));
}

bool IsSignedInteger(ElementType elementType) {
  MLIRContext ctx;
  Type type = getElementType(ctx, elementType);

  // Note that this is not the same as `type.isSignedInteger()`. Signed integers
  // are not used in StableHLO.
  return type.isSignlessInteger() && !IsBoolean(elementType);
}

bool IsUnsignedInteger(ElementType elementType) {
  MLIRContext ctx;
  return getElementType(ctx, elementType).isUnsignedInteger() &&
         !IsBoolean(elementType);
}

RankedTensorType makeTensorType(MLIRContext& ctx, ArrayRef<int64_t> shape,
                                ElementType elementType) {
  return makeTensorType(ctx, shape, getElementType(ctx, elementType));
}

RankedTensorType makeTensorType(MLIRContext& ctx, ArrayRef<int64_t> shape,
                                Type elementType) {
  return RankedTensorType::get(shape, elementType);
}

//////////////////////
// Builders - Constant Literals
//////////////////////

namespace detail {

APFloat toAPFloat(double val, FloatType floatType) {
  llvm::APFloat apf(val);
  const auto& fltSemantics = floatType.getFloatSemantics();
  auto roundingMode = APFloat::rmNearestTiesToEven;
  bool losesInfo;
  apf.convert(fltSemantics, roundingMode, &losesInfo);
  return apf;
}

}  // namespace detail

}  // namespace mlir
