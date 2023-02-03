/* Copyright 2023 The StableHLO Authors.

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

#ifndef STABLEHLO_DIALECT_VHLO_DIALECT_H
#define STABLEHLO_DIALECT_VHLO_DIALECT_H

#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"

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

  /// Return a Version representing the current dialect version.
  static Version getCurrentVersion() { return Version(0, 4, 0); }

  /// Return a Version representing the minimum supported dialect version.
  static Version getMinimumVersion() { return Version(0, 3, 0); }
};

}  // namespace vhlo
}  // end namespace mlir

#endif  // STABLEHLO_DIALECT_VHLO_DIALECT_H
