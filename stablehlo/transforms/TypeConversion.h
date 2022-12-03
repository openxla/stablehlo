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

#include "llvm/Support/Debug.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/dialect/VhloOps.h"

#define DEBUG_TYPE "compat-passes"

namespace mlir {
namespace vhlo {

class VersionedTypeConverterBase : public TypeConverter {
 public:
  VersionedTypeConverterBase() : TypeConverter() {
    addConversion([](Type t) -> Type { return t; });
    addConversion([&](TupleType type) -> Type {
      SmallVector<Type> convertedTypes;
      if (failed(convertTypes(type.getTypes(), convertedTypes))) return {};
      return TupleType::get(type.getContext(), convertedTypes);
    });
    addConversion([&](RankedTensorType type) -> Type {
      auto encoding = type.getEncoding();
      if (!encoding) return type;
      if (isSourceDialect(encoding.getDialect())) {
        auto convertedEncoding = convertEncoding(encoding);
        if (!convertedEncoding) return {};
        return RankedTensorType::get(type.getShape(), type.getElementType(),
                                     convertedEncoding);
      }
      return type;
    });
  };

  virtual ~VersionedTypeConverterBase() = default;

  // Checks whether the given dialect is the source dialect of the type
  // conversion (e.g. StableHLO for StablehloToVhloTypeConverter).
  virtual bool isSourceDialect(Dialect& dialect) = 0;

  virtual Attribute convertEncoding(Attribute attr) = 0;
};

class StablehloToVhloTypeConverter : public VersionedTypeConverterBase {
 public:
  StablehloToVhloTypeConverter() : VersionedTypeConverterBase() {
    addConversion([](stablehlo::TokenType token) -> Type {
      LLVM_DEBUG(llvm::dbgs() << "Converting TokenType\n");
      return TokenType::get(token.getContext());
    });
  }

  bool isSourceDialect(Dialect& dialect) final {
    return dialect.getNamespace() ==
           stablehlo::StablehloDialect::getDialectNamespace();
  }

  Attribute convertEncoding(Attribute attr) final {
    LLVM_DEBUG(llvm::dbgs() << "Converting encoding.\n");
    LLVM_DEBUG(llvm::dbgs() << attr);
    if (auto stablehloAttr =
            attr.dyn_cast_or_null<stablehlo::TypeExtensionsAttr>()) {
      LLVM_DEBUG(llvm::dbgs() << "Matched StableHLO encoding.\n");
      return vhlo::TypeExtensionsAttr::get(stablehloAttr.getContext(),
                                           stablehloAttr.getBounds());
    }
    // All encodings should be supported.
    return {};
  }
};

class VhloToStablehloTypeConverter : public VersionedTypeConverterBase {
 public:
  VhloToStablehloTypeConverter() : VersionedTypeConverterBase() {
    addConversion([](vhlo::TokenType token) -> Type {
      LLVM_DEBUG(llvm::dbgs() << "Converting TokenType\n");
      return stablehlo::TokenType::get(token.getContext());
    });
  }

  bool isSourceDialect(Dialect& dialect) final {
    return dialect.getNamespace() == vhlo::VhloDialect::getDialectNamespace();
  }

  Attribute convertEncoding(Attribute attr) final {
    if (auto vhloAttr = attr.dyn_cast_or_null<vhlo::TypeExtensionsAttr>()) {
      return stablehlo::TypeExtensionsAttr::get(vhloAttr.getContext(),
                                                vhloAttr.getBounds());
    }
    // All encodings should be supported.
    return attr;
  }
};

class VhloToVersionConverter : public VersionedTypeConverterBase {
 public:
  VhloToVersionConverter() : VersionedTypeConverterBase() {
    addConversion([](stablehlo::TokenType token) -> Type {
      LLVM_DEBUG(llvm::dbgs() << "Converting TokenType\n");
      return TokenType::get(token.getContext());
    });
  }

  bool isSourceDialect(Dialect& dialect) final {
    return dialect.getNamespace() == vhlo::VhloDialect::getDialectNamespace();
  }

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
