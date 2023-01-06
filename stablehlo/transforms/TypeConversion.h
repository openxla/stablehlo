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

#ifndef STABLEHLO_TRANSFORMS_TYPECONVERSION_H
#define STABLEHLO_TRANSFORMS_TYPECONVERSION_H

#include "llvm/ADT/APFloat.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Quant/QuantTypes.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/dialect/VhloOps.h"

#define DEBUG_TYPE "compat-passes"

namespace mlir {
namespace vhlo {

class VersionedTypeConverterBase : public TypeConverter {
 public:
  VersionedTypeConverterBase() : TypeConverter(){};

  virtual ~VersionedTypeConverterBase() = default;

  virtual Attribute convertEncoding(Attribute attr) = 0;
};

class StablehloToVhloTypeConverter : public VersionedTypeConverterBase {
 public:
  StablehloToVhloTypeConverter() : VersionedTypeConverterBase() {
    addConversion([](Type type) -> Type {
      if (type.getDialect().getNamespace() == "vhlo") {
        return type;
      }
      LLVM_DEBUG(llvm::dbgs() << "Invalid type: " << type << '\n');
      return Type();
    });
    addConversion([](stablehlo::TokenType token) -> Type {
      return TokenType::get(token.getContext());
    });

    // Forked Types
    addConversion([&](BFloat16Type type) {
      return BFloat16V1Type::get(type.getContext());
    });
    addConversion([&](ComplexType type) {
      return ComplexV1Type::get(type.getContext(),
                                convertType(type.getElementType()));
    });
    addConversion([&](Float16Type type) {
      return Float16V1Type::get(type.getContext());
    });
    addConversion([&](Float32Type type) {
      return Float32V1Type::get(type.getContext());
    });
    addConversion([&](Float64Type type) {
      return Float64V1Type::get(type.getContext());
    });
    addConversion(
        [&](IndexType type) { return IndexV1Type::get(type.getContext()); });
    addConversion([&](IntegerType type) {
      return IntegerV1Type::get(type.getContext(), type);
    });
    addConversion([&](RankedTensorType type) -> Type {
      auto encoding = type.getEncoding();
      auto convertedEncoding = encoding ? convertEncoding(encoding) : encoding;
      auto convertedElementType = convertType(type.getElementType());
      if ((encoding && !convertedEncoding) || !convertedElementType) return {};
      return RankedTensorV1Type::get(type.getContext(), type.getShape(),
                                     convertedElementType, convertedEncoding);
    });
    addConversion([&](TupleType type) -> Type {
      SmallVector<Type> convertedTypes;
      if (failed(convertTypes(type.getTypes(), convertedTypes))) return {};
      return vhlo::TupleV1Type::get(type.getContext(), convertedTypes);
    });
    addConversion([&](quant::UniformQuantizedType type) -> Type {
      Type storage = convertType(type.getStorageType());
      Type expressed = convertType(type.getExpressedType());
      if (!storage || !expressed) return {};
      return vhlo::UniformQuantizedV1Type::get(
          type.getContext(), type.getFlags(), storage, expressed,
          APFloat(type.getScale()), type.getZeroPoint(),
          type.getStorageTypeMin(), type.getStorageTypeMax());
    });
    addConversion([&](UnrankedTensorType type) -> Type {
      auto convertedElementType = convertType(type.getElementType());
      if (!convertedElementType) return {};  // unsupported element type
      return UnrankedTensorV1Type::get(type.getContext(), convertedElementType);
    });
    addConversion([&](shape::WitnessType type) -> Type {
      return vhlo::WitnessV1Type::get(type.getContext());
    });
  }

  bool isTargetDialect(Dialect& dialect) {
    return dialect.getNamespace() == vhlo::VhloDialect::getDialectNamespace();
  }

  Attribute convertEncoding(Attribute attr) final {
    LLVM_DEBUG(llvm::dbgs() << "Converting encoding.\n" << attr << '\n');

    // Must be VHLO encoding, or convertible to VHLO encoding.
    if (isTargetDialect(attr.getDialect())) return attr;
    if (auto stablehloAttr =
            attr.dyn_cast_or_null<stablehlo::TypeExtensionsAttr>()) {
      return vhlo::TypeExtensionsAttr::get(stablehloAttr.getContext(),
                                           stablehloAttr.getBounds());
    }

    // Was not VHLO encoding, or convertible.
    return {};
  }
};

class VhloToStablehloTypeConverter : public VersionedTypeConverterBase {
 public:
  VhloToStablehloTypeConverter() : VersionedTypeConverterBase() {
    addConversion([](Type type) -> Type { return type; });
    addConversion([](vhlo::TokenType token) -> Type {
      LLVM_DEBUG(llvm::dbgs() << "Converting TokenType\n");
      return stablehlo::TokenType::get(token.getContext());
    });

    // Forked types
    addConversion([&](BFloat16V1Type type) {
      return BFloat16Type::get(type.getContext());
    });
    addConversion([&](ComplexV1Type type) {
      return ComplexType::get(convertType(type.getElementType()));
    });
    addConversion([&](Float16V1Type type) {
      return Float16Type::get(type.getContext());
    });
    addConversion([&](Float32V1Type type) {
      return Float32Type::get(type.getContext());
    });
    addConversion([&](Float64V1Type type) {
      return Float64Type::get(type.getContext());
    });
    addConversion(
        [&](IndexV1Type type) { return IndexType::get(type.getContext()); });
    addConversion([&](IntegerV1Type type) { return type.getValue(); });
    addConversion([&](RankedTensorV1Type type) -> Type {
      auto encoding = type.getEncoding();
      auto convertedEncoding = encoding ? convertEncoding(encoding) : encoding;
      auto convertedElementType = convertType(type.getElementType());
      if ((encoding && !convertedEncoding) || !convertedElementType) return {};
      return RankedTensorType::get(type.getShape(), convertedElementType,
                                   convertedEncoding);
    });
    addConversion([&](TupleV1Type type) -> Type {
      SmallVector<Type> convertedTypes;
      if (failed(convertTypes(type.getTypes(), convertedTypes))) return {};
      return TupleType::get(type.getContext(), convertedTypes);
    });
    addConversion([&](UniformQuantizedV1Type type) -> Type {
      Type storage = convertType(type.getStorageType());
      Type expressed = convertType(type.getExpressedType());
      if (!storage || !expressed) return {};
      return quant::UniformQuantizedType::get(
          type.getFlags(), storage, expressed,
          type.getScale().convertToDouble(), type.getZeroPoint(),
          type.getStorageTypeMin(), type.getStorageTypeMax());
    });
    addConversion([&](UnrankedTensorV1Type type) -> Type {
      auto convertedElementType = convertType(type.getElementType());
      if (!convertedElementType) return {};  // unsupported element type
      return UnrankedTensorType::get(convertedElementType);
    });
    addConversion([&](WitnessV1Type type) -> Type {
      return shape::WitnessType::get(type.getContext());
    });
  }

  Attribute convertEncoding(Attribute attr) final {
    if (auto vhloAttr = attr.dyn_cast_or_null<vhlo::TypeExtensionsAttr>()) {
      return stablehlo::TypeExtensionsAttr::get(vhloAttr.getContext(),
                                                vhloAttr.getBounds());
    }
    // All encodings supported in StableHLO.
    return attr;
  }
};

class VhloToVersionConverter : public VersionedTypeConverterBase {
 public:
  VhloToVersionConverter() : VersionedTypeConverterBase() {
    addConversion([](Type type) -> Type {
      if (type.getDialect().getNamespace() == "vhlo") {
        return type;
      }
      LLVM_DEBUG(llvm::dbgs() << "Invalid type: " << type << '\n');
      return Type();
    });

    addConversion([](stablehlo::TokenType token) -> Type {
      LLVM_DEBUG(llvm::dbgs() << "Converting TokenType\n");
      return TokenType::get(token.getContext());
    });
  }

  // All encodings from VHLO -> VHLO are valid.
  Attribute convertEncoding(Attribute attr) final { return attr; }
};

// Complements conversion patterns with boilerplate that makes sure `func.func`,
// `func.call` and `func.return` ops which involve illegal types get converted
// to use legal types.
void registerFuncOpsForTypeConversion(ConversionTarget& target,
                                      RewritePatternSet& patterns,
                                      TypeConverter& converter);
}  // namespace vhlo
}  // namespace mlir

#endif  // STABLEHLO_TRANSFORMS_MAPSTABLEHLOTOVHLO_H
