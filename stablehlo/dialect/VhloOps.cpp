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

#include <cstdint>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Quant/QuantOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "stablehlo/dialect/AssemblyFormat.h"
#include "stablehlo/dialect/VhloBytecode.h"

namespace mlir {
namespace vhlo {

namespace {
// Helper functions for VHLO verifiers
template <typename TypeOrAttr>
bool isFromVhlo(TypeOrAttr t) {
  return t.getDialect().getNamespace() == VhloDialect::getDialectNamespace();
}

template <typename TypeOrAttr>
bool allFromVhlo(ArrayRef<TypeOrAttr> range) {
  return llvm::all_of(range, isFromVhlo<TypeOrAttr>);
}

Type convertTypeToBuiltinForPrint(Type type) {
  struct VhloToBuiltinPrintConverter : VhloTypeConverter {
    VhloToBuiltinPrintConverter() : VhloTypeConverter() {
      addVhloToBuiltinConversions();
    }
    Attribute convertEncoding(Attribute attr) override { return attr; }
  };
  VhloToBuiltinPrintConverter conv;
  return conv.convertType(type);
}

Type convertTypeToVhloForParse(Type type) {
  struct BuiltinToVhloParseConverter : VhloTypeConverter {
    BuiltinToVhloParseConverter() : VhloTypeConverter() {
      addBuiltinToVhloConversions();
    }
    Attribute convertEncoding(Attribute attr) override { return attr; }
  };
  BuiltinToVhloParseConverter conv;
  return conv.convertType(type);
}

}  // namespace
// Helper functions for VHLO printers and parsers
static void printAttributeArray(AsmPrinter& os, ArrayRef<Attribute> arrayAttr) {
  os << '[' << arrayAttr << ']';
}

// Parse attributes in brackets: [#vhlo.attr, #vhlo.attr]
ParseResult parseAttributeArray(AsmParser& parser,
                                SmallVector<Attribute>& arrayAttr) {
  ArrayAttr array;
  if (failed(parser.parseAttribute(array))) {
    return failure();
  }
  arrayAttr.append(array.begin(), array.end());
  return success();
}

void IntegerV1Attr::print(mlir::AsmPrinter& p) const {
  p << '<'
    << IntegerAttr::get(convertTypeToBuiltinForPrint(getType()), getValue())
    << '>';
}

Attribute IntegerV1Attr::parse(AsmParser& parser, mlir::Type) {
  IntegerAttr attr;
  if (failed(parser.parseLess()) || failed(parser.parseAttribute(attr)) ||
      failed(parser.parseGreater())) {
    return IntegerV1Attr();
  }
  return IntegerV1Attr::get(parser.getContext(),
                            convertTypeToVhloForParse(attr.getType()),
                            attr.getValue());
}

static void printAttributeDictionary(
    AsmPrinter& os, ArrayRef<std::pair<Attribute, Attribute>> values) {
  os << '{';
  for (auto nvp : values) {
    os << nvp.first << " = " << nvp.second;
  }
  os << '}';
}

// Parse array of NVPs in braces: {key = value, key = value}
ParseResult parseAttributeDictionary(
    AsmParser& parser, SmallVector<std::pair<Attribute, Attribute>>& values) {
  auto parseEle = [&]() {
    Attribute name;
    Attribute value;
    if (failed(parser.parseAttribute(name)) || failed(parser.parseEqual()) ||
        failed(parser.parseAttribute(value))) {
      return failure();
    }
    values.push_back({name, value});
    return success();
  };
  if (failed(parser.parseCommaSeparatedList(AsmParser::Delimiter::Braces,
                                            parseEle))) {
    return failure();
  }
  return success();
}

// Print function using: @name(arg : type, ...) -> (res_type...) { body_ops }
void printFunctionBody(OpAsmPrinter& p, Operation*, Attribute name,
                       Region& region, Attribute funcType) {
  p.printSymbolName(name.cast<StringV1Attr>().getValue());
  p << '(';
  llvm::interleaveComma(region.getArguments(), p,
                        [&](auto arg) { p.printRegionArgument(arg); });
  p << ") -> (";
  auto fnType = funcType.cast<TypeV1Attr>().getValue().cast<FunctionV1Type>();
  llvm::interleaveComma(fnType.getOutputs(), p,
                        [&](auto res) { p.printType(res); });
  p << ") ";
  p.printRegion(region, false, true, true);
}

// Parse function using: @name(arg : type, ...) -> (res_type...) { body_ops }
ParseResult parseFunctionBody(OpAsmParser& parser, Attribute& name,
                              Region& region, Attribute& funcType) {
  StringAttr strName;
  SmallVector<OpAsmParser::Argument> args;
  SmallVector<Type> inputTypes;
  SmallVector<Type> resultTypes;
  if (failed(parser.parseSymbolName(strName)) ||
      failed(
          parser.parseArgumentList(args, AsmParser::Delimiter::Paren, true)) ||
      failed(parser.parseArrowTypeList(resultTypes)) ||
      failed(parser.parseRegion(region, args))) {
    return failure();
  }
  name = StringV1Attr::get(parser.getContext(), strName.getValue());
  for (OpAsmParser::Argument arg : args) {
    inputTypes.push_back(arg.type);
  }
  funcType = TypeV1Attr::get(
      parser.getContext(),
      FunctionV1Type::get(parser.getContext(), inputTypes, resultTypes));

  return success();
}

void TensorV1Attr::print(mlir::AsmPrinter& p) const {
  p << '<'
    << DenseIntOrFPElementsAttr::getFromRawBuffer(
           convertTypeToBuiltinForPrint(getType()), getData())
    << '>';
}

// Parse tensor elements using DenseIntOrFPElementsAttr printing.
Attribute TensorV1Attr::parse(AsmParser& parser, mlir::Type) {
  DenseIntOrFPElementsAttr attr;
  if (failed(parser.parseLess()) || failed(parser.parseAttribute(attr)) ||
      failed(parser.parseGreater())) {
    return TensorV1Attr();
  }
  return TensorV1Attr::get(parser.getContext(),
                           convertTypeToVhloForParse(attr.getType()),
                           attr.getRawData());
}

void printEscapedString(AsmPrinter& p, llvm::StringRef value) {
  p << "\"";
  llvm::printEscapedString(value, p.getStream());
  p << "\"";
}

ParseResult parseEscapedString(AsmParser& parser, std::string& value) {
  return parser.parseString(&value);
}

}  // namespace vhlo
}  // namespace mlir

// Include order matters
#include "stablehlo/dialect/VhloAttrInterfaces.cpp.inc"
#include "stablehlo/dialect/VhloEnums.cpp.inc"
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
  addVhloTypes();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "stablehlo/dialect/VhloAttrs.cpp.inc"
      >();
}

void VhloDialect::addVhloTypes() {
  // Idiomatically, this functionality is expressed as shown below:
  //
  //     addTypes<
  //   #define GET_TYPEDEF_LIST
  //   #include "stablehlo/dialect/VhloTypeDefs.cpp.inc"
  //         >();
  //
  // However, Dialect::addTypes doesn't work for our situation where we want
  // to decouple the vhlo_ops and vhlo_types targets. VhloTypeDefs.h.inc only
  // includes forward declarations of TypeStorage structs, and that's not enough
  // for Dialect::addTypes to compile.
  //
  // Therefore, we work around by introducing this function and then
  // reimplementing Dialect::addTypes as shown below.
  addTypesWithoutRegistering<
#define GET_TYPEDEF_LIST
#include "stablehlo/dialect/VhloTypeDefs.cpp.inc"
      >();
  registerVhloTypes(getContext());
}

Type VhloDialect::parseType(DialectAsmParser& parser) const {
  StringRef dataType;
  Type type;
  auto parseResultOpt = parseVhloType(parser, &dataType, type);
  if (parseResultOpt.has_value() && succeeded(*parseResultOpt)) return type;
  parser.emitError(parser.getNameLoc()) << "unknown vhlo type: " << dataType;
  return nullptr;
}

void VhloDialect::printType(Type type, DialectAsmPrinter& os) const {
  if (succeeded(printVhloType(type, os))) return;
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
