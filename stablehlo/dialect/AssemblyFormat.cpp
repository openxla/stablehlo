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

#include "stablehlo/dialect/AssemblyFormat.h"

#include <cstdint>
#include <string>

#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Regex.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"

namespace mlir {
namespace hlo {
//===----------------------------------------------------------------------===//
// Generic Type Printer and Parser
//===----------------------------------------------------------------------===//
namespace {
// Utility function, used by printSelectOpType and
// printSameOperandsAndResultType. Given a FunctionType, assign the types
// to operands and results, erroring if any mismatch in number of operands
// or results occurs.
ParseResult assignFromFunctionType(OpAsmParser& parser, llvm::SMLoc loc,
                                   ArrayRef<Type*> operands, Type& result,
                                   FunctionType& fnType) {
  assert(fnType);
  if (fnType.getInputs().size() != operands.size())
    return parser.emitError(loc)
           << operands.size() << " operands present, but expected "
           << fnType.getInputs().size();

  // Set operand types to function input types
  for (auto [operand, input] : llvm::zip(operands, fnType.getInputs()))
    *operand = input;

  // Set result type
  if (fnType.getResults().size() != 1)
    return parser.emitError(loc, "expected single output");
  result = fnType.getResults()[0];

  return success();
}
}  // namespace

namespace detail {
void printSameOperandsAndResultTypeImpl(OpAsmPrinter& p, Operation* op,
                                        TypeRange operands, Type result) {
  // Handle zero operand types `() -> a` prints `a`
  if (operands.empty()) {
    p.printType(result);
    return;
  }

  // Handle all same type `(a,a,...) -> a` prints `a`
  bool allSameType =
      llvm::all_of(operands, [&result](auto t) { return t == result; });
  if (allSameType) {
    p.printType(result);
    return;
  }

  // Fall back to generic
  p.printFunctionalType(op);
}

ParseResult parseSameOperandsAndResultTypeImpl(OpAsmParser& parser,
                                               ArrayRef<Type*> operands,
                                               Type& result) {
  llvm::SMLoc loc = parser.getCurrentLocation();
  Type type;
  if (parser.parseType(type)) return failure();

  // Handle if function type, all operand types did not match result type.
  if (auto fnType = type.dyn_cast<FunctionType>())
    return assignFromFunctionType(parser, loc, operands, result, fnType);

  // Handle bare types. ` : type` indicating all input/output types match.
  for (Type* t : operands) *t = type;
  result = type;
  return success();
}
}  // namespace detail

void printVariadicSameOperandsAndResultType(OpAsmPrinter& p, Operation* op,
                                            OperandRange operands,
                                            TypeRange opTypes, Type result) {
  return detail::printSameOperandsAndResultTypeImpl(p, op, opTypes, result);
}

ParseResult parseVariadicSameOperandsAndResultType(
    OpAsmParser& parser,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand>& operands,
    SmallVectorImpl<Type>& opTypes, Type& result) {
  // Insert a type for each operand. Need to do this since passing the type of
  // a variadic op gives no indication of how many operands were provided.
  opTypes.resize(operands.size());

  // Make a pointer list to the operands
  SmallVector<Type*> typePtrs;
  typePtrs.reserve(opTypes.size());
  for (Type& t : opTypes) typePtrs.push_back(&t);

  return detail::parseSameOperandsAndResultTypeImpl(parser, typePtrs, result);
}

void printTupleOpType(OpAsmPrinter& p, Operation*, TypeRange operands,
                      Type result) {
  p.printType(result);
}

ParseResult parseTupleOpType(OpAsmParser& parser,
                             SmallVectorImpl<Type>& operands, Type& result) {
  // Result type must be tuple type.
  llvm::SMLoc loc = parser.getCurrentLocation();
  if (parser.parseType(result)) return failure();

  auto tupType = result.dyn_cast<TupleType>();
  if (!tupType) return parser.emitError(loc, "expected tuple type");

  // Assign operand types to tuple types
  llvm::append_range(operands, tupType.getTypes());
  return success();
}

void printPairwiseOpType(OpAsmPrinter& p, Operation*, TypeRange operands,
                         TypeRange results) {
  llvm::interleaveComma(operands, p);
}

ParseResult parsePairwiseOpType(OpAsmParser& parser,
                                SmallVectorImpl<Type>& operands,
                                SmallVectorImpl<Type>& results) {
  llvm::SMLoc loc = parser.getCurrentLocation();
  if (parser.parseTypeList(operands))
    return parser.emitError(loc, "expected type list");
  results = operands;
  return success();
}

void printVariadicOperandWithAttribute(OpAsmPrinter& p, Operation*,
                                       OperandRange operands) {
  llvm::interleaveComma(operands, p);
  p << ",";
}

ParseResult parseVariadicOperandWithAttribute(
    OpAsmParser& parser,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand>& operands) {
  // Parse operands as well as trailing commas. Stops when first non-ssa value
  // seen.
  OpAsmParser::UnresolvedOperand operand;
  auto resultOpt = parser.parseOptionalOperand(operand);
  while (resultOpt.has_value() && succeeded(resultOpt.value())) {
    operands.push_back(operand);
    if (failed(parser.parseComma())) return failure();
    resultOpt = parser.parseOptionalOperand(operand);
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Operation Printers and Parsers
//===----------------------------------------------------------------------===//

void printComplexOpType(OpAsmPrinter& p, Operation* op, Type lhs, Type rhs,
                        Type result) {
  Type realType = createRealType(result.cast<TensorType>());

  if (lhs != realType || rhs != realType) {
    p.printFunctionalType(op);
    return;
  }

  p.printType(result);
}

ParseResult parseComplexOpType(OpAsmParser& parser, Type& lhs, Type& rhs,
                               Type& result) {
  llvm::SMLoc loc = parser.getCurrentLocation();
  Type type;
  if (failed(parser.parseType(type))) return failure();

  // Handle if function type, all operand types did not match result type.
  if (auto fnType = type.dyn_cast<FunctionType>())
    return assignFromFunctionType(parser, loc, {&lhs, &rhs}, result, fnType);

  // Otherwise, operand type is inferred from complex type
  auto tensorType = type.dyn_cast<TensorType>();
  if (!tensorType || !tensorType.getElementType().isa<ComplexType>())
    return parser.emitError(loc, "expected tensor with complex element type");

  // Assign LHS and RHS to inferred type
  Type realType = createRealType(type.cast<TensorType>());
  lhs = rhs = realType;
  result = type;
  return success();
}

void printSelectOpType(OpAsmPrinter& p, Operation* op, Type pred, Type onTrue,
                       Type onFalse, Type result) {
  // Print functional type if true/false branches don't match return type.
  if (onTrue != result || onFalse != result) {
    p.printFunctionalType(op);
    return;
  }

  // Print pred type and result type
  p << pred << ", " << result;
}

ParseResult parseSelectOpType(OpAsmParser& parser, Type& pred, Type& onTrue,
                              Type& onFalse, Type& result) {
  llvm::SMLoc loc = parser.getCurrentLocation();
  SmallVector<Type> types;
  if (parser.parseTypeList(types)) return failure();

  // Error handling for invalid types
  // Fail if not two types, or single functional type
  bool isValidType = (types.size() == 2 ||
                      (types.size() == 1 && types[0].isa<FunctionType>()));
  if (!isValidType)
    return parser.emitError(loc,
                            "expected functional type or list of two types");

  // stablehlo.select %0, %1 : <pred_type>, <op_and_result_type>
  if (types.size() == 2) {
    pred = types[0];
    onTrue = onFalse = result = types[1];
    return success();
  }

  // stablehlo.select %0, %1 : (<op_types> ...) -> <result_type>
  auto fnType = types[0].cast<FunctionType>();
  return assignFromFunctionType(parser, loc, {&pred, &onTrue, &onFalse}, result,
                                fnType);
}

//===----------------------------------------------------------------------===//
// Attribute Printers and Parsers
//===----------------------------------------------------------------------===//

void printDenseI64Array(OpAsmPrinter& p, Operation* op,
                        DenseIntElementsAttr attr) {
  if (attr.getType().getRank() != 1)
    llvm::report_fatal_error("printDenseI64Array only supports rank-1 arrays");
  auto values = llvm::to_vector(attr.getValues<int64_t>());
  DenseI64ArrayAttr arrayAttr =
      DenseI64ArrayAttr::get(op->getContext(), values);
  arrayAttr.print(p);
}

ParseResult parseDenseI64Array(OpAsmParser& parser,
                               DenseIntElementsAttr& attr) {
  DenseI64ArrayAttr arrayAttr = DenseI64ArrayAttr::parse(parser, Type{})
                                    .dyn_cast_or_null<DenseI64ArrayAttr>();
  if (!arrayAttr) return failure();

  ArrayRef<int64_t> data = arrayAttr.asArrayRef();
  RankedTensorType type =
      RankedTensorType::get(data.size(), parser.getBuilder().getI64Type());
  attr = DenseIntElementsAttr::get(type, data);
  return success();
}

ParseResult dimSizeFromString(AsmParser& parser, int64_t& result) {
  if (succeeded(parser.parseOptionalQuestion())) {
    result = ShapedType::kDynamic;
    return success();
  }
  return parser.parseInteger(result);
}

std::string dimSizeToString(int64_t dimSize) {
  if (hlo::isDynamicDimSize(dimSize)) return "?";
  return std::to_string(dimSize);
}

template <typename Stream>
void printDimSizes(Stream& stream, ArrayRef<int64_t> dimSizes) {
  stream << '[';
  llvm::interleaveComma(dimSizes, stream, [&](int64_t dimSize) {
    stream << dimSizeToString(dimSize);
  });
  stream << ']';
}

std::string dimSizesToString(ArrayRef<int64_t> dimSizes) {
  std::string buffer;
  llvm::raw_string_ostream os(buffer);
  printDimSizes(os, dimSizes);
  return buffer;
}

void printDimSizes(AsmPrinter& p, ArrayRef<int64_t> dimSizes) {
  printDimSizes<AsmPrinter>(p, dimSizes);
}

FailureOr<SmallVector<int64_t>> parseDimSizes(AsmParser& parser) {
  SmallVector<int64_t> dimSizes;
  if (failed(parseDimSizes(parser, dimSizes))) return failure();
  return dimSizes;
}

ParseResult parseDimSizes(AsmParser& parser, SmallVector<int64_t>& dimSizes) {
  return parser.parseCommaSeparatedList(AsmParser::Delimiter::Square, [&]() {
    return dimSizeFromString(parser, dimSizes.emplace_back());
  });
}

// Print attributes as e#m#
void printExponentMantissa(AsmPrinter& p, Operation*, IntegerAttr exponent,
                           IntegerAttr mantissa) {
  p << 'e';
  p.printAttributeWithoutType(exponent);
  p << 'm';
  p.printAttributeWithoutType(mantissa);
}

// Parse e#m# as exponent=# and mantissa=#
ParseResult parseExponentMantissa(AsmParser& parser, IntegerAttr& exponent,
                                  IntegerAttr& mantissa) {
  llvm::SMLoc loc = parser.getCurrentLocation();
  llvm::StringRef expMan;
  if (parser.parseKeyword(&expMan)) return failure();

  // Validate format e#m#
  llvm::Regex expManRegex("^e([0-9]+)m([0-9]+)$");
  llvm::SmallVector<llvm::StringRef> matches;
  if (!expManRegex.match(expMan, &matches))
    return parser.emitError(loc,
                            "expected exponent mantissa in format e#m#, saw ")
           << expMan;

  // Parse off digits of exp/man
  assert(matches.size() == 3);  // matches[0] is entire regex match.
  llvm::StringRef expS = matches[1];
  llvm::StringRef manS = matches[2];

  // Parse as base 10 integer strings
  int exp, mant;
  if (expS.getAsInteger(/*radix=*/10, exp))
    return parser.emitError(loc, "unable to parse exponent '")
           << expS.str() << "'";
  if (manS.getAsInteger(/*radix=*/10, mant))
    return parser.emitError(loc, "unable to parse mantissa '")
           << manS.str() << "'";

  exponent = parser.getBuilder().getI32IntegerAttr(exp);
  mantissa = parser.getBuilder().getI32IntegerAttr(mant);
  return success();
}

void printCustomCallTarget(AsmPrinter& p, Operation*, StringAttr target) {
  p.printSymbolName(target.getValue());
}

ParseResult parseCustomCallTarget(AsmParser& parser, StringAttr& target) {
  return parser.parseSymbolName(target);
}

void printTypeExtensions(BoundedAttrInterface attr, DialectAsmPrinter& os) {
  os << "bounds<";
  llvm::interleaveComma(attr.getBounds(), os,
                        [&](int64_t bound) { os << dimSizeToString(bound); });
  os << ">";
}

Attribute parseTypeExtensions(HloDialectInterface* dialect,
                              DialectAsmParser& parser) {
  SmallVector<int64_t> bounds;
  if (failed(parser.parseCommaSeparatedList(
          AsmParser::Delimiter::LessGreater,
          [&]() { return dimSizeFromString(parser, bounds.emplace_back()); })))
    return nullptr;
  return dialect->createTypeExtensions(bounds);
}

}  // namespace hlo
}  // namespace mlir
