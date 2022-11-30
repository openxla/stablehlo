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

#include "llvm/Support/Debug.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo/dialect/VhloOps.h"
#include "stablehlo/dialect/StablehloOps.h"

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
    // FIXME: Do I need to do anything with ranked tensor / encoding here?
  };
};

class StablehloToVhloTypeConverter : public VersionedTypeConverterBase {
 public:
  StablehloToVhloTypeConverter() : VersionedTypeConverterBase() {
    addConversion([](stablehlo::TokenType token) -> Type {
      LLVM_DEBUG(llvm::dbgs() << "Converting TokenType\n");
      return TokenType::get(token.getContext());
    });
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
};

// Complements conversion patterns with boilerplate that makes sure `func.func`,
// `func.call` and `func.return` ops which involve illegal types get converted
// to use legal types.
void registerFuncOpsForTypeConversion(ConversionTarget& target,
                                      RewritePatternSet& patterns,
                                      TypeConverter& converter);
}  // namespace vhlo
}  // namespace mlir