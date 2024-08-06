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

#include <cassert>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <tuple>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "stablehlo/dialect/Base.h"

#define DEBUG_TYPE "hlo-assembly"

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
  if (auto fnType = dyn_cast<FunctionType>(type))
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

void printConstantOp(OpAsmPrinter& p, Operation* op, ElementsAttr value) {
  assert(op->getNumResults() == 1);
  // If not all types are the same, use generic form.
  if (value.getType() != op->getResultTypes().front()) {
    p.printGenericOp(op, /*printOpName=*/false);
    return;
  }

  p.printOptionalAttrDict(op->getAttrs(), /*elidedAttrs=*/{"value"});
  p << ' ';
  p.printStrippedAttrOrType(value);
}

ParseResult parseConstantOp(OpAsmParser& parser, OperationState& result) {
  // Parse the generic form.
  if (succeeded(parser.parseOptionalLParen())) {
    if (parser.parseRParen()) return failure();
    // Parse optional properties
    if (succeeded(parser.parseOptionalLess()) &&
        (failed(parser.parseAttribute(result.propertiesAttr)) ||
         failed(parser.parseGreater())))
      return failure();

    // Parse optional attributes
    if (parser.parseOptionalAttrDict(result.attributes)) return failure();

    // Parse type signature
    if (parser.parseColon() || parser.parseLParen() || parser.parseRParen() ||
        parser.parseArrow())
      return failure();
    Type resultTy;
    if (parser.parseType(resultTy)) return failure();
    result.addTypes(resultTy);
    return success();
  }

  ElementsAttr valueAttr;
  if (parser.parseOptionalAttrDict(result.attributes)) return failure();
  if (parser.parseCustomAttributeWithFallback(valueAttr, Type{}, "value",
                                              result.attributes))
    return failure();
  result.addTypes(valueAttr.getType());
  return success();
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

  auto tupType = dyn_cast<TupleType>(result);
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

void printComplexOpType(OpAsmPrinter& p, Operation* op, ShapedType lhs,
                        ShapedType rhs, ShapedType result) {
  Type realType = createRealType(result);

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
  if (auto fnType = dyn_cast<FunctionType>(type))
    return assignFromFunctionType(parser, loc, {&lhs, &rhs}, result, fnType);

  // Otherwise, operand type is inferred from complex type
  auto shapedType = dyn_cast<ShapedType>(type);
  if (!shapedType || !isa<ComplexType>(shapedType.getElementType()))
    return parser.emitError(loc, "expected tensor with complex element type");

  // Assign LHS and RHS to inferred type
  Type realType = createRealType(shapedType);
  lhs = rhs = realType;
  result = type;
  return success();
}

namespace {
void createArgs(ArrayRef<OpAsmParser::UnresolvedOperand> operands,
                ArrayRef<Type> types,
                SmallVector<OpAsmParser::Argument>& args) {
  for (auto argAndType : llvm::zip(operands, types)) {
    auto& arg = args.emplace_back();
    arg.ssaName = std::get<0>(argAndType);
    arg.type = std::get<1>(argAndType);
  }
}

Operation* createReturn(OpBuilder& builder, Dialect* dialect, Location loc,
                        ResultRange operands) {
  auto returnOpName = dialect->getNamespace() + ".return";
  OperationState returnOpState(loc, returnOpName.str());
  returnOpState.operands.append(operands.begin(), operands.end());
  return builder.create(returnOpState);
}

bool hasSameOperandAndResultTypes(Operation& op) {
  Type expected;
  if (op.getNumResults() != 0) expected = op.getResult(0).getType();
  if (op.getNumOperands() != 0) expected = op.getOperand(0).getType();
  if (!expected) return false;

  auto typeMatch = [&](Type actual) { return actual == expected; };
  return llvm::all_of(op.getOperandTypes(), typeMatch) &&
         llvm::all_of(op.getResultTypes(), typeMatch);
}

// Helper method for E2
bool isCommutativeNoRegionMatchingDialect(OperationName innerOp,
                                          StringRef reduceOpDialect) {
  auto innerOpDialect = innerOp.getDialect();
  return innerOpDialect && innerOpDialect->getNamespace() == reduceOpDialect &&
         innerOp.hasTrait<mlir::OpTrait::NOperands<2>::Impl>() &&
         innerOp.hasTrait<mlir::OpTrait::OneResult>() &&
         (innerOp.hasTrait<mlir::hlo::OpTrait::IsCommutative>() ||
          innerOp.hasTrait<mlir::OpTrait::IsCommutative>()) &&
         innerOp.hasTrait<mlir::OpTrait::ZeroRegions>();
}

// Checks the following eligibility criteria for compact printing of reduce:
// E1. The reduce-op wraps a single inner-op in the associated region.
// E2. The single operation is a commutative binary-op from the dialect, zero
//     region, producing single result such that the operands and result all
//     have the same type.
// E3. The reduce-op consist of at least one input-operand; The operand-types of
//     inner-op should be derived trivially from the element-type of reduce-op's
//     first input-operand.
// E4. The arguments of the region's only basic block are forwarded perfectly
//     to inner-op's operands.
// E5. The single operation result is perfectly forwarded to the reduce op
//     return.
static bool isReduceEligibleForCompactPrint(Operation* op, ValueRange inputs,
                                            Region& body) {
  // Check E1.
  LLVM_DEBUG(llvm::dbgs() << "Checking ReduceOp compact print E1\n");
  auto& block = body.front();
  if (!hasSingleElement(block.without_terminator())) return false;

  Operation& innerOp = *block.begin();

  // Check E2.
  LLVM_DEBUG(llvm::dbgs() << "Checking ReduceOp compact print E2\n");
  if (innerOp.getDialect() != op->getDialect()) return false;

  if (!isCommutativeNoRegionMatchingDialect(innerOp.getName(),
                                            op->getDialect()->getNamespace()) ||
      !hasSameOperandAndResultTypes(innerOp))
    return false;

  // Check E3.
  LLVM_DEBUG(llvm::dbgs() << "Checking ReduceOp compact print E3\n");
  if (inputs.empty()) return false;

  auto elemType = cast<ShapedType>(inputs[0].getType()).getElementType();
  auto expectedInnerOpType = RankedTensorType::get(/*shape=*/{}, elemType);
  if (innerOp.getOperands()[0].getType() != expectedInnerOpType) return false;

  // Check E4.
  LLVM_DEBUG(llvm::dbgs() << "Checking ReduceOp compact print E4\n");
  if (!llvm::equal(block.getArguments(), innerOp.getOperands())) return false;

  // Check E5.
  LLVM_DEBUG(llvm::dbgs() << "Checking ReduceOp compact print E5\n");
  auto retOp = block.getTerminator();
  if (retOp->getName().stripDialect() != "return") return false;

  return llvm::equal(innerOp.getResults(), retOp->getOperands());
}
}  // namespace

void printReduceOp(OpAsmPrinter& p, Operation* op, ValueRange inputs,
                   ArrayRef<int64_t> dimensions, Region& body) {
  int numOperandPairs = op->getNumOperands() / 2;
  auto printOpAndInit = [&](int opId) {
    p << "(" << op->getOperand(opId)
      << " init: " << op->getOperand(opId + numOperandPairs) << ")";
  };
  // Print the pairs of operands under the form:
  //   (%arg0 init: %arg3), (%arg1 init: %arg4), (%arg2 init: %arg5)
  llvm::interleaveComma(llvm::seq<int>(0, numOperandPairs), p, printOpAndInit);

  // Print compact if eligible
  bool printCompact = isReduceEligibleForCompactPrint(op, inputs, body);
  if (printCompact) {
    Operation& innerOp = body.front().front();
    p << " applies ";
    p.printKeywordOrString(innerOp.getName().getStringRef());
  }
  p << " across dimensions = [";
  llvm::interleaveComma(dimensions, p);
  p << "]";
  p.printOptionalAttrDict(op->getAttrs(), {"dimensions"});
  p << " : ";
  p.printFunctionalType(op);
  if (!printCompact) {
    p.printNewline();
    p << " reducer";
    // Print the pairs of block operands under the form:
    //   (%arg0_elt, %arg0_acc) (%arg1_elt, %arg1_acc):
    Block& reducer = body.front();
    auto printReducerOpAndInit = [&](int opId) {
      p << "(";
      p.printRegionArgument(reducer.getArgument(opId));
      p << ", ";
      p.printRegionArgument(reducer.getArgument(opId + numOperandPairs));
      p << ") ";
    };
    llvm::for_each(llvm::seq<int>(0, numOperandPairs), printReducerOpAndInit);
    p << ' ';
    p.printRegion(body, /*printEntryBlockArgs=*/false);
  }
}

ParseResult parseReduceOp(
    OpAsmParser& parser, OperationState& result,
    std::function<Attribute(OpBuilder&, ArrayRef<int64_t>)> createDimensions) {
  llvm::SMLoc loc = parser.getCurrentLocation();
  Location currLocation = parser.getEncodedSourceLoc(loc);

  // Parse the operands of reduce-op, this is a list of pair under the form:
  //   (%arg0 init: %arg3), (%arg1 init: %arg4), (%arg2 init: %arg5)
  // Each input to reduce is paired with its init value, even though in memory
  // they are stored with the input first and the init values after.
  SmallVector<OpAsmParser::UnresolvedOperand, 2> operands;
  SmallVector<OpAsmParser::UnresolvedOperand, 2> initOperands;
  auto parseEle = [&]() -> ParseResult {
    if (parser.parseOptionalLParen()) return success();
    if (parser.parseOperand(operands.emplace_back()) ||
        parser.parseKeyword("init") || parser.parseColon() ||
        parser.parseOperand(initOperands.emplace_back()) ||
        parser.parseRParen())
      return failure();
    return success();
  };
  if (failed(parser.parseCommaSeparatedList(parseEle))) return failure();
  operands.append(initOperands);

  // Check if we are parsing the compact version of reduce-op:
  // stablehlo.reduce applies <inner-op> across dimensions = [...] : <func-type>
  // else parse the "region-based" variant.
  if (failed(parser.parseOptionalKeyword("applies"))) {
    // Parse the inner-op dimensions, reduce-op's function-type and
    // optional location.
    SmallVector<int64_t> dimensions;
    auto parseDim = [&]() -> ParseResult {
      if (parser.parseInteger(dimensions.emplace_back())) return failure();
      return success();
    };

    FunctionType reduceOpFnType;
    if (parser.parseKeyword("across") || parser.parseKeyword("dimensions") ||
        parser.parseEqual() ||
        parser.parseCommaSeparatedList(AsmParser::Delimiter::Square,
                                       parseDim) ||
        parser.parseOptionalAttrDict(result.attributes) ||
        parser.parseColon() || parser.parseType(reduceOpFnType) ||
        parser.parseKeyword("reducer"))
      return failure();
    OpBuilder builder(parser.getContext());
    result.addAttribute("dimensions", createDimensions(builder, dimensions));

    // Parse the "reducer" region now.
    SmallVector<OpAsmParser::UnresolvedOperand, 2> reducerOperands;
    SmallVector<OpAsmParser::UnresolvedOperand, 2> reducerInitOperands;
    SmallVector<Type, 2> reducerTypes;
    SmallVector<Type, 2> reducerInitTypes;
    SmallVector<std::optional<Location>, 2> reducerLocs;
    SmallVector<std::optional<Location>, 2> reducerInitLocs;
    auto parseBlockOperand =
        [&](SmallVectorImpl<OpAsmParser::UnresolvedOperand>& operands,
            SmallVectorImpl<Type>& types,
            SmallVectorImpl<std::optional<Location>>& locs) -> ParseResult {
      if (parser.parseOperand(operands.emplace_back(),
                              /*allowResultNumber=*/false) ||
          parser.parseColon() || parser.parseType(types.emplace_back()) ||
          parser.parseOptionalLocationSpecifier(locs.emplace_back()))
        return failure();
      return success();
    };
    while (succeeded(parser.parseOptionalLParen())) {
      if (parseBlockOperand(reducerOperands, reducerTypes, reducerLocs) ||
          parser.parseComma() ||
          parseBlockOperand(reducerInitOperands, reducerInitTypes,
                            reducerInitLocs) ||
          parser.parseRParen())
        return failure();
    }
    reducerOperands.append(reducerInitOperands);
    reducerTypes.append(reducerInitTypes);
    reducerLocs.append(reducerInitLocs);
    result.addTypes(reduceOpFnType.getResults());
    SmallVector<OpAsmParser::Argument> reducerArgs;
    createArgs(reducerOperands, reducerTypes, reducerArgs);

    // Derive the SSA-values for reduce-op's operands and parse the region, and
    // the optional trailing location.
    std::optional<Location> trailingLoc;
    if (parser.resolveOperands(operands, reduceOpFnType.getInputs(), loc,
                               result.operands) ||
        parser.parseRegion(*result.addRegion(), reducerArgs))
      return failure();
    // Set the individual block arguments.
    for (auto argAndLoc :
         llvm::zip(result.regions.front()->front().getArguments(), reducerLocs))
      if (std::get<1>(argAndLoc))
        std::get<0>(argAndLoc).setLoc(std::get<1>(argAndLoc).value());
    result.location = trailingLoc.value_or(currLocation);
    return success();
  }

  // Parse the inner-op name and check if the contract on inner-op
  // mentioned in "isEligibleForCompactPrint::E2" for pretty-printing is met.
  FailureOr<OperationName> innerOpNameInfo = parser.parseCustomOperationName();
  if (failed(innerOpNameInfo)) return failure();

  StringRef innerOpName = innerOpNameInfo->getStringRef();
  StringRef reduceOpDialect = result.name.getDialectNamespace();
  LLVM_DEBUG(llvm::dbgs() << "Reduce: " << reduceOpDialect << "\n");
  LLVM_DEBUG(llvm::dbgs() << "Inner: "
                          << innerOpNameInfo->getDialect()->getNamespace()
                          << "\n");
  if (!isCommutativeNoRegionMatchingDialect(*innerOpNameInfo, reduceOpDialect))
    return parser.emitError(
        loc,
        "expected the inner-op to be a commutative binary-op from "
        "the " +
            reduceOpDialect +
            " dialect, with zero region, producing single result");

  // Parse the inner-op dimensions, reduce-op's function-type and
  // optional location.
  SmallVector<int64_t> dimensions;
  auto parseDim = [&]() -> ParseResult {
    if (parser.parseInteger(dimensions.emplace_back())) return failure();
    return success();
  };

  std::optional<Location> explicitLoc;
  FunctionType reduceOpFnType;
  if (parser.parseKeyword("across") || parser.parseKeyword("dimensions") ||
      parser.parseEqual() ||
      parser.parseCommaSeparatedList(AsmParser::Delimiter::Square, parseDim) ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.parseType(reduceOpFnType) ||
      parser.parseOptionalLocationSpecifier(explicitLoc))
    return failure();

  if (!reduceOpFnType || reduceOpFnType.getInputs().empty()) {
    if (!reduceOpFnType) return parser.emitError(loc, "expected function type");
    return parser.emitError(loc,
                            "input types missing in reduce-op function type");
  }

  // If location of reduce-op is explicitly provided, then use it; Else use
  // the parser's current location.
  Location reduceOpLoc = explicitLoc.value_or(currLocation);

  // Derive the SSA-values for reduce-op's operands.
  if (parser.resolveOperands(operands, reduceOpFnType.getInputs(), loc,
                             result.operands))
    return failure();

  // Derive the type of inner-op from that of reduce-op's input operand.
  auto innerOpType = RankedTensorType::get(
      /*shape=*/{}, getElementTypeOrSelf(reduceOpFnType.getInput(0)));

  // Add a region for reduce-op.
  Region& region = *result.addRegion();

  // Create a basic-block inside reduce-op's region.
  Block& block = region.emplaceBlock();
  auto lhs = block.addArgument(innerOpType, reduceOpLoc);
  auto rhs = block.addArgument(innerOpType, reduceOpLoc);

  // Create and insert an "inner-op" operation in the block.
  OpBuilder builder(parser.getContext());
  builder.setInsertionPointToStart(&block);

  OperationState innerOpState(reduceOpLoc, innerOpName);
  innerOpState.operands.push_back(lhs);
  innerOpState.operands.push_back(rhs);
  innerOpState.addTypes(innerOpType);

  Operation* innerOp = builder.create(innerOpState);

  // Insert a return statement in the block returning the inner-op's result.
  createReturn(builder, innerOp->getDialect(), innerOp->getLoc(),
               innerOp->getResults());

  // Populate the reduce-op operation-state with result-type, location, and
  // dimension attribute.
  result.addTypes(reduceOpFnType.getResults());
  result.location = innerOp->getLoc();
  result.addAttribute("dimensions", createDimensions(builder, dimensions));
  return success();
}

void printSelectOpType(OpAsmPrinter& p, Operation* op, ShapedType pred,
                       ShapedType onTrue, ShapedType onFalse,
                       ShapedType result) {
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
  bool isValidType =
      (types.size() == 2 || (types.size() == 1 && isa<FunctionType>(types[0])));
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
  auto fnType = cast<FunctionType>(types[0]);
  return assignFromFunctionType(parser, loc, {&pred, &onTrue, &onFalse}, result,
                                fnType);
}

void printWhileOp(OpAsmPrinter& p, Operation* op, Region& cond, Region& body) {
  p << '(';
  llvm::interleaveComma(llvm::zip(body.getArguments(), op->getOperands()), p,
                        [&](auto zip) {
                          p.printOperand(std::get<0>(zip));
                          p << " = ";
                          p.printOperand(std::get<1>(zip));
                        });
  p << ")";
  if (op->getNumOperands()) {
    p << " : ";
    llvm::interleaveComma(op->getOperandTypes(), p);
  }
  p.printOptionalAttrDictWithKeyword(op->getAttrs());
  p.printNewline();
  p << " cond ";
  p.printRegion(cond, /*printEntryBlockArgs=*/false);
  p << " do ";
  p.printRegion(body, /*printEntryBlockArgs=*/false);
}

ParseResult parseWhileOp(OpAsmParser& parser, OperationState& result) {
  llvm::SMLoc loc = parser.getCurrentLocation();
  // Parse the operands of the while: these are of the form:
  //   %iter_arg = %init_val
  // where %iter_arg is the name of the block argument in the cond/body blocks
  // and %init_val is the actual operand.
  SmallVector<OpAsmParser::UnresolvedOperand> operands;
  SmallVector<OpAsmParser::UnresolvedOperand> iterArgs;
  if (parser.parseLParen()) return failure();
  do {
    if (succeeded(parser.parseOptionalRParen())) break;
    OpAsmParser::UnresolvedOperand operand, iterArg;
    if (parser.parseOperand(iterArg) || parser.parseEqual() ||
        parser.parseOperand(operand))
      return failure();
    iterArgs.push_back(iterArg);
    operands.push_back(operand);
    if (succeeded(parser.parseOptionalRParen())) break;
    if (failed(parser.parseComma())) return failure();
  } while (true);
  if (!operands.empty()) {
    if (parser.parseColon() || parser.parseTypeList(result.types))
      return failure();
  }
  SmallVector<OpAsmParser::Argument> args;
  createArgs(iterArgs, result.types, args);
  if (parser.resolveOperands(operands, result.types, loc, result.operands) ||
      parser.parseOptionalAttrDictWithKeyword(result.attributes) ||
      parser.parseKeyword("cond") ||
      parser.parseRegion(*result.addRegion(), args) ||
      parser.parseKeyword("do") ||
      parser.parseRegion(*result.addRegion(), args))
    return failure();
  return success();
}

//===----------------------------------------------------------------------===//
// Attribute Printers and Parsers
//===----------------------------------------------------------------------===//

void printSliceRanges(OpAsmPrinter& p, Operation* op,
                      ArrayRef<int64_t> startIndices,
                      ArrayRef<int64_t> limitIndices,
                      ArrayRef<int64_t> strides) {
  p << "[";
  // Let's be safe if we're printing invalid IR somehow: this can't be parsed
  // back!
  if (startIndices.size() != limitIndices.size() ||
      startIndices.size() != strides.size()) {
    p << "start_indices: ";
    llvm::interleaveComma(startIndices, p);
    p << ", limit_indices: ";
    llvm::interleaveComma(limitIndices, p);
    p << ", strides: ";
    llvm::interleaveComma(strides, p);
    p << "]";
    return;
  }

  llvm::interleaveComma(llvm::zip(startIndices, limitIndices, strides), p,
                        [&](std::tuple<int64_t, int64_t, int64_t> pack) {
                          auto [start, limit, stride] = pack;
                          p << start << ":" << limit;
                          if (stride != 1) {
                            p << ":" << stride;
                          }
                        });
  p << "]";
}

ParseResult parseSliceRanges(OpAsmParser& parser,
                             DenseI64ArrayAttr& startIndices,
                             DenseI64ArrayAttr& limitIndices,
                             DenseI64ArrayAttr& strides) {
  if (parser.parseLSquare()) return failure();
  // Parse groups of comma-separated: `start`:`limit`[:`stride`]
  // If the stride isn't provided it'll be 1.
  SmallVector<int64_t> start, limit, stride;
  if (failed(parser.parseOptionalRSquare())) {
    do {
      start.emplace_back();
      limit.emplace_back();
      if (parser.parseInteger(start.back()) || parser.parseColon() ||
          parser.parseInteger(limit.back()))
        return failure();
      if (parser.parseOptionalColon()) {
        stride.push_back(1);
      } else {
        stride.emplace_back();
        if (parser.parseInteger(stride.back())) return failure();
      }
      if (succeeded(parser.parseOptionalRSquare())) break;
      if (failed(parser.parseComma())) return failure();
    } while (1);
  }

  startIndices = parser.getBuilder().getDenseI64ArrayAttr(start);
  limitIndices = parser.getBuilder().getDenseI64ArrayAttr(limit);
  strides = parser.getBuilder().getDenseI64ArrayAttr(stride);

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
