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

#ifndef STABLEHLO_DIALECT_ASSEMBLYFORMAT_H
#define STABLEHLO_DIALECT_ASSEMBLYFORMAT_H

#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"

namespace mlir {
namespace hlo {

//===----------------------------------------------------------------------===//
// Generic Type Printers and Parsers
//===----------------------------------------------------------------------===//

// Declarative `custom<SameOperandsAndResultType>(...)` implementation:
// Pretty print for ops with many operands, but one result type, simplifies
// print if all operand types match the result type.
//
// Example:
//   custom<SameOperandsAndResultType>(type($result), type($operand1),
//   type($operand2))
//
//   Generic:
//     %0 = "stablehlo.op"(%0, %1) : (tensor<i1>, tensor<i1>) -> tensor<i1>
//   Custom:
//     %0 = stablehlo.op(%0, %1) : tensor<i1>
//
// Falls back to `printFunctionalType` if all operands do not match result
// type.
//
// Note that `type($result)` is the first argument, this is done because the
// behavior of trailing parameter packs is easily understandable.
namespace detail {
void printSameOperandsAndResultTypeImpl(OpAsmPrinter& p, Operation* op,
                                        TypeRange operands, Type result);

ParseResult parseSameOperandsAndResultTypeImpl(OpAsmParser& parser,
                                               ArrayRef<Type*> operands,
                                               Type& result);
}  // namespace detail

template <class... OpTypes>
void printSameOperandsAndResultType(OpAsmPrinter& p, Operation* op,
                                    OpTypes... types) {
  static_assert(sizeof...(types) > 0);  // Must be non empty, must have result
  SmallVector<Type> typesVec{types...};
  ArrayRef<Type> typesRef = makeArrayRef(typesVec);
  return detail::printSameOperandsAndResultTypeImpl(
      p, op, typesRef.drop_back(1), typesRef.back());
}

template <class... OpTypes>
ParseResult parseSameOperandsAndResultType(OpAsmParser& parser,
                                           OpTypes&... types) {
  static_assert(sizeof...(types) > 0);  // Must be non empty, must have result
  SmallVector<Type*> typesVec{&types...};
  ArrayRef<Type*> typesRef = makeArrayRef(typesVec);
  return detail::parseSameOperandsAndResultTypeImpl(
      parser, typesRef.drop_back(1), *typesRef.back());
}

void printVariadicSameOperandsAndResultType(OpAsmPrinter& p, Operation* op,
                                            OperandRange operands,
                                            TypeRange opTypes, Type result);

ParseResult parseVariadicSameOperandsAndResultType(
    OpAsmParser& parser,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand>& operands,
    SmallVectorImpl<Type>& opTypes, Type& result);

// TuplesOp - only print result type. Operand type is trivially inferrable.
//
// Inferring operand types from tuple type:
//  %3 = stablehlo.tuple %1, %2 : tuple<tensor<i1>, tensor<f32>>
//    %1 : tensor<i1>
//    %2 : tensor<f32>
//    %3 : tuple<tensor<i1>, tensor<f32>>
void printTupleOpType(OpAsmPrinter& p, Operation*, TypeRange operands,
                      Type result);

ParseResult parseTupleOpType(OpAsmParser& parser,
                             SmallVectorImpl<Type>& operands, Type& result);

// PairwiseOps - only print result type. Operand types are trivially
// inferrable.
//
// Inferring operand types for pairwise ops:
//  %3, %4 = stablehlo.operation %1, %2 : tensor<i1>, tensor<f32>
//    %1 : tensor<i1>
//    %2 : tensor<f32>
//    %3 : tensor<i1>
//    %4 : tensor<f32>
void printPairwiseOpType(OpAsmPrinter& p, Operation*, TypeRange operands,
                         TypeRange results);

ParseResult parsePairwiseOpType(OpAsmParser& parser,
                                SmallVectorImpl<Type>& operands,
                                SmallVectorImpl<Type>& results);

// Variadic operands with attributes - Need to provide custom parser since
// the built-in operand list parser parses the attribute expecting an SSA value
// and errors.
//
// %0 = stablehlo.operation %arg0, ..., %argN, attr = value
void printVariadicOperandWithAttribute(OpAsmPrinter& p, Operation*,
                                       OperandRange operands);

ParseResult parseVariadicOperandWithAttribute(
    OpAsmParser& parser,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand>& operands);

//===----------------------------------------------------------------------===//
// Operation Printers and Parsers
//===----------------------------------------------------------------------===//

void printComplexOpType(OpAsmPrinter& p, Operation* op, Type lhs, Type rhs,
                        Type result);

ParseResult parseComplexOpType(OpAsmParser& parser, Type& lhs, Type& rhs,
                               Type& result);

void printSelectOpType(OpAsmPrinter& p, Operation* op, Type pred, Type onTrue,
                       Type onFalse, Type result);

ParseResult parseSelectOpType(OpAsmParser& parser, Type& pred, Type& onTrue,
                              Type& onFalse, Type& result);

//===----------------------------------------------------------------------===//
// Attribute Printers and Parsers
//===----------------------------------------------------------------------===//

void printExponentMantissa(AsmPrinter& p, Operation*, IntegerAttr exponent,
                           IntegerAttr mantissa);

ParseResult parseExponentMantissa(AsmParser& parser, IntegerAttr& exponent,
                                  IntegerAttr& mantissa);

}  // namespace hlo
}  // namespace mlir

#endif  // STABLEHLO_DIALECT_ASSEMBLYFORMAT_H
