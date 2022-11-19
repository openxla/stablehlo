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

#include "stablehlo/compatibility/dialect/VersionedStablehloOps.h"
#include "stablehlo/dialect/AssemblyFormat.h"

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/TypeUtilities.h"

// Include order matters
#include "stablehlo/compatibility/dialect/VersionedStablehloEnums.cpp.inc"
#define GET_ATTRDEF_CLASSES
#include "stablehlo/compatibility/dialect/VersionedStablehloAttrs.cpp.inc"
#include "stablehlo/compatibility/dialect/VersionedStablehloOpInterfaces.cpp.inc"
#define GET_OP_CLASSES
#include "stablehlo/compatibility/dialect/VersionedStablehloOps.cpp.inc"

namespace mlir {
namespace versionedhlo {

using mlir::hlo::printIntArray;

//===----------------------------------------------------------------------===//
// StableHLO Dialect Constructor
//===----------------------------------------------------------------------===//

VersionedhloDialect::VersionedhloDialect(MLIRContext* context)
    : Dialect(getDialectNamespace(), context,
              TypeID::get<VersionedhloDialect>()) {
  addOperations<
#define GET_OP_LIST
#include "stablehlo/compatibility/dialect/VersionedStablehloOps.cpp.inc"
      >();
  // TODO (gleasonk): addBytecodeInterface(this);
  addTypes<TokenType>();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "stablehlo/compatibility/dialect/VersionedStablehloAttrs.cpp.inc"
      >();
}

Type VersionedhloDialect::parseType(DialectAsmParser& parser) const {
  StringRef dataType;
  if (parser.parseKeyword(&dataType)) return Type();

  if (dataType == "token") return TokenType::get(getContext());
  parser.emitError(parser.getNameLoc())
      << "unknown stablehlo type: " << dataType;
  return nullptr;
}

void VersionedhloDialect::printType(Type type, DialectAsmPrinter& os) const {
  if (type.isa<TokenType>()) {
    os << "token";
    return;
  }
  os << "<unknown stablehlo type>";
}

// Entry point for Attribute parsing, TableGen generated code will handle the
// dispatch to the individual classes.
Attribute VersionedhloDialect::parseAttribute(DialectAsmParser& parser,
                                              Type type) const {
  StringRef attrTag;
  Attribute attr;
  auto parseResult = generatedAttributeParser(parser, &attrTag, type, attr);
  if (parseResult.has_value()) return attr;
  parser.emitError(parser.getNameLoc(), "unknown stablehlo attribute");
  return Attribute();
}

// Entry point for Attribute printing, TableGen generated code will handle the
// dispatch to the individual classes.
void VersionedhloDialect::printAttribute(Attribute attr,
                                         DialectAsmPrinter& os) const {
  LogicalResult result = generatedAttributePrinter(attr, os);
  (void)result;
  assert(succeeded(result));
}

}  // namespace versionedhlo
}  // namespace mlir
