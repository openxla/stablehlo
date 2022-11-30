/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.
   Copyright 2022 The StableHLO Authors.

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

#ifndef STABLEHLO_COMPATIBILITY_VHLO_OPS_H
#define STABLEHLO_COMPATIBILITY_VHLO_OPS_H

#include "stablehlo/compatibility/dialect/VersionNumber.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Regex.h"
#include "mlir/Dialect/Quant/QuantTypes.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"

// Include order matters.
#include "stablehlo/compatibility/dialect/VhloEnums.h.inc"
#define GET_ATTRDEF_CLASSES
#include "stablehlo/compatibility/dialect/VhloAttrs.h.inc"

namespace mlir {
namespace vhlo {

class VhloDialect : public Dialect {
 public:
  explicit VhloDialect(MLIRContext *context);
  static StringRef getDialectNamespace() { return "vhlo"; }

  // Parses a type registered to this dialect.
  Type parseType(DialectAsmParser &parser) const override;

  // Prints a type registered to this dialect.
  void printType(Type type, DialectAsmPrinter &os) const override;

  // Parses an attribute registered to this dialect.
  Attribute parseAttribute(DialectAsmParser &parser, Type type) const override;

  // Prints an attribute registered to this dialect.
  void printAttribute(Attribute attr, DialectAsmPrinter &os) const override;

  // Return a value represenging the max supported dialect
  static llvm::StringRef getCurrentDialectVersion() {
    return "0.2.0";
  }

  static llvm::StringRef getMinimumDialectVersion() {
    return "0.0.0";
  }
};

class TokenType : public Type::TypeBase<TokenType, Type, TypeStorage> {
 public:
  using Base::Base;
};

}  // namespace vhlo
}  // end namespace mlir

#include "stablehlo/compatibility/dialect/VhloOpInterfaces.h.inc"
#define GET_OP_CLASSES
#include "stablehlo/compatibility/dialect/VhloOps.h.inc"

#endif  // STABLEHLO_COMPATIBILITY_VHLO_OPS_H