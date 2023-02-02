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
#include "stablehlo/transforms/VhloBuiltinTypeConversion.h"

#define DEBUG_TYPE "compat-passes"

namespace mlir {
namespace vhlo {

class StablehloToVhloTypeConverter : public VhloBuiltinTypeConverter {
 public:
  StablehloToVhloTypeConverter() : VhloBuiltinTypeConverter() {
    addConversion([](Type type) -> Type {
      if (type.getDialect().getNamespace() ==
          vhlo::VhloDialect::getDialectNamespace()) {
        return type;
      }
      LLVM_DEBUG(llvm::dbgs() << "Invalid type: " << type << '\n');
      return Type();
    });
    addConversion([](stablehlo::TokenType token) -> Type {
      return TokenType::get(token.getContext());
    });
    addBuiltinToVhloConversions();
  }

  Attribute convertEncoding(Attribute attr) final {
    LLVM_DEBUG(llvm::dbgs() << "Converting encoding.\n" << attr << '\n');

    // Must be VHLO encoding, or convertible to VHLO encoding.

    if (attr.getDialect().getNamespace() ==
        vhlo::VhloDialect::getDialectNamespace())
      return attr;
    if (auto stablehloAttr =
            attr.dyn_cast_or_null<stablehlo::TypeExtensionsAttr>()) {
      return vhlo::TypeExtensionsV1Attr::get(stablehloAttr.getContext(),
                                             stablehloAttr.getBounds());
    }

    // Was not VHLO encoding, or convertible.
    return {};
  }
};

class VhloToStablehloTypeConverter : public VhloBuiltinTypeConverter {
 public:
  VhloToStablehloTypeConverter() : VhloBuiltinTypeConverter() {
    addConversion([](Type type) -> Type { return type; });
    addConversion([](vhlo::TokenType token) -> Type {
      LLVM_DEBUG(llvm::dbgs() << "Converting TokenType\n");
      return stablehlo::TokenType::get(token.getContext());
    });
    addVhloToBuiltinConversions();
  }

  Attribute convertEncoding(Attribute attr) final {
    if (auto vhloAttr = attr.dyn_cast_or_null<vhlo::TypeExtensionsV1Attr>()) {
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
      if (type.getDialect().getNamespace() ==
          vhlo::VhloDialect::getDialectNamespace()) {
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

#undef DEBUG_TYPE

#endif  // STABLEHLO_TRANSFORMS_MAPSTABLEHLOTOVHLO_H
