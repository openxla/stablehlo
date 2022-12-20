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

#include "stablehlo/dialect/VhloOps.h"

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "stablehlo/dialect/AssemblyFormat.h"
#include "stablehlo/dialect/VhloBytecode.h"

// Include order matters
#include "stablehlo/dialect/VhloTypeInterfaces.cpp.inc"
#define GET_TYPEDEF_CLASSES
#include "stablehlo/dialect/VhloAttrInterfaces.cpp.inc"
#include "stablehlo/dialect/VhloEnums.cpp.inc"
#include "stablehlo/dialect/VhloTypeDefs.cpp.inc"
#define GET_ATTRDEF_CLASSES
#include "stablehlo/dialect/VhloAttrs.cpp.inc"
#include "stablehlo/dialect/VhloOpInterfaces.cpp.inc"
#define GET_OP_CLASSES
#include "stablehlo/dialect/VhloOps.cpp.inc"

namespace mlir {
namespace vhlo {

//===----------------------------------------------------------------------===//
// StableHLO Dialect Constructor
//===----------------------------------------------------------------------===//

VhloDialect::VhloDialect(MLIRContext* context)
    : Dialect(getDialectNamespace(), context, TypeID::get<VhloDialect>()) {
  addOperations<
#define GET_OP_LIST
#include "stablehlo/dialect/VhloOps.cpp.inc"
      >();
  addBytecodeInterface(this);
  addTypes<
#define GET_TYPEDEF_LIST
#include "stablehlo/dialect/VhloTypeDefs.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "stablehlo/dialect/VhloAttrs.cpp.inc"
      >();
}

Type VhloDialect::parseType(DialectAsmParser& parser) const {
  StringRef dataType;
  Type type;
  auto parseResultOpt = generatedTypeParser(parser, &dataType, type);
  if (parseResultOpt.has_value() && succeeded(*parseResultOpt)) {
    return type;
  }
  parser.emitError(parser.getNameLoc()) << "unknown vhlo type: " << dataType;
  return nullptr;
}

void VhloDialect::printType(Type type, DialectAsmPrinter& os) const {
  if (succeeded(generatedTypePrinter(type, os))) {
    return;
  }
  os << "<unknown vhlo type>";
}

// Entry point for Attribute parsing, TableGen generated code will handle the
// dispatch to the individual classes.
Attribute VhloDialect::parseAttribute(DialectAsmParser& parser,
                                      Type type) const {
  StringRef attrTag;
  Attribute attr;
  auto parseResult = generatedAttributeParser(parser, &attrTag, type, attr);
  if (parseResult.has_value()) return attr;
  parser.emitError(parser.getNameLoc(), "unknown vhlo attribute");
  return Attribute();
}

// Entry point for Attribute printing, TableGen generated code will handle the
// dispatch to the individual classes.
void VhloDialect::printAttribute(Attribute attr, DialectAsmPrinter& os) const {
  LogicalResult result = generatedAttributePrinter(attr, os);
  (void)result;
  assert(succeeded(result));
}

}  // namespace vhlo
}  // namespace mlir
