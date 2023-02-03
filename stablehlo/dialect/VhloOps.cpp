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
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Quant/QuantOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "stablehlo/dialect/AssemblyFormat.h"
#include "stablehlo/dialect/VhloBytecode.h"
#include "stablehlo/transforms/TypeConversion.h"

namespace mlir {
namespace vhlo {

static void printEncoding(AsmPrinter& os, Attribute encoding) {
  if (!encoding) return;
  os << ", " << encoding;
}

ParseResult parseEncoding(AsmParser& parser, FailureOr<Attribute>& encoding) {
  Attribute attr;
  if (failed(parser.parseOptionalComma())) {
    encoding = attr;
    return success();
  }
  if (failed(parser.parseAttribute(attr))) return failure();
  encoding = attr;
  return success();
}

static void printTensorShape(AsmPrinter& os, ArrayRef<int64_t> dimSizes) {
  if (dimSizes.empty()) return;
  for (int64_t dimSize : dimSizes) {
    os << hlo::dimSizeToString(dimSize) << 'x';
  }
}

ParseResult parseTensorShape(AsmParser& parser,
                             FailureOr<SmallVector<int64_t>>& dimSizes) {
  SmallVector<int64_t> sizes;
  if (failed(parser.parseDimensionList(sizes))) {
    return failure();
  }
  dimSizes = sizes;
  return success();
}

static void printAttributeArray(AsmPrinter& os, ArrayRef<Attribute> arrayAttr) {
  os << '[' << arrayAttr << ']';
}

ParseResult parseAttributeArray(AsmParser& parser,
                                FailureOr<SmallVector<Attribute>>& arrayAttr) {
  ArrayAttr array;
  if (failed(parser.parseAttribute(array))) {
    return failure();
  }
  SmallVector<Attribute> values(array.begin(), array.end());
  arrayAttr = values;
  return success();
}

static void printFloatValue(const APFloat& apValue, AsmPrinter& os) {
  // We would like to output the FP constant value in exponential notation,
  // but we cannot do this if doing so will lose precision.  Check here to
  // make sure that we only output it in exponential format if we can parse
  // the value back and get the same value.
  bool isInf = apValue.isInfinity();
  bool isNaN = apValue.isNaN();
  if (!isInf && !isNaN) {
    SmallString<128> strValue;
    apValue.toString(strValue, /*FormatPrecision=*/6, /*FormatMaxPadding=*/0,
                     /*TruncateZero=*/false);

    // Check to make sure that the stringized number is not some string like
    // "Inf" or NaN, that atof will accept, but the lexer will not.  Check
    // that the string matches the "[-+]?[0-9]" regex.
    assert(((strValue[0] >= '0' && strValue[0] <= '9') ||
            ((strValue[0] == '-' || strValue[0] == '+') &&
             (strValue[1] >= '0' && strValue[1] <= '9'))) &&
           "[-+]?[0-9] regex does not match!");

    // Parse back the stringized version and check that the value is equal
    // (i.e., there is no precision loss).
    if (APFloat(apValue.getSemantics(), strValue).bitwiseIsEqual(apValue)) {
      os << strValue;
      return;
    }

    // If it is not, use the default format of APFloat instead of the
    // exponential notation.
    strValue.clear();
    apValue.toString(strValue);

    // Make sure that we can parse the default form as a float.
    if (strValue.str().contains('.')) {
      os << strValue;
      return;
    }
  }

  // Print special values in hexadecimal format. The sign bit should be included
  // in the literal.
  SmallVector<char, 16> str;
  APInt apInt = apValue.bitcastToAPInt();
  apInt.toString(str, /*Radix=*/16, /*Signed=*/false,
                 /*formatAsCLiteral=*/true);
  os << str;
}

void DenseIntOrFPElementsV1Attr::print(mlir::AsmPrinter& p) const {
  VhloToStablehloTypeConverter conv;
  p << '<'
    << DenseIntOrFPElementsAttr::getFromRawBuffer(conv.convertType(getType()),
                                                  getRawData())
    << '>';
}

Attribute DenseIntOrFPElementsV1Attr::parse(AsmParser& parser, mlir::Type) {
  StablehloToVhloTypeConverter conv;
  DenseIntOrFPElementsAttr attr;
  if (failed(parser.parseLess()) || failed(parser.parseAttribute(attr)) ||
      failed(parser.parseGreater())) {
    return DenseIntOrFPElementsV1Attr();
  }
  return DenseIntOrFPElementsV1Attr::get(
      parser.getContext(), conv.convertType(attr.getType()), attr.getRawData());
}

}  // namespace vhlo
}  // namespace mlir

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
  // context->loadDialect<shape::ShapeDialect>();
  // context->loadDialect<quant::QuantizationDialect>();
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
