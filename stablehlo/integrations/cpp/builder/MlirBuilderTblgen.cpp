/* Copyright 2025 The StableHLO Authors.

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

#include <cassert>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Main.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"
#include "mlir/Support/IndentedOstream.h"
#include "mlir/Support/LLVM.h"
#include "mlir/TableGen/Argument.h"
#include "mlir/TableGen/Attribute.h"
#include "mlir/TableGen/Class.h"
#include "mlir/TableGen/Format.h"
#include "mlir/TableGen/Operator.h"
#include "mlir/TableGen/Region.h"

using llvm::raw_ostream;
using llvm::RecordKeeper;
using llvm::StringRef;
using mlir::tblgen::Attribute;
using mlir::tblgen::FmtContext;
using mlir::tblgen::Method;
using mlir::tblgen::MethodBody;
using mlir::tblgen::MethodParameter;
using mlir::tblgen::NamedAttribute;
using mlir::tblgen::Operator;

namespace mlir {
namespace {

enum class ActionType {
  GenBuilderHeader,
  GenBuilderImpl,
  GenBuilderDocs,
};

static llvm::cl::opt<ActionType> action(
    llvm::cl::desc("action to perform"),
    llvm::cl::values(clEnumValN(ActionType::GenBuilderHeader,
                                "gen-builder-decls", "")),
    llvm::cl::values(clEnumValN(ActionType::GenBuilderImpl, "gen-builder-defs",
                                "")),
    llvm::cl::values(clEnumValN(ActionType::GenBuilderDocs, "gen-builder-docs",
                                "")));

LogicalResult skipOperation(const Operator& op, StringRef reason) {
  llvm::errs() << "Skipping " << op.getCppClassName() << ": " << reason << "\n";
  return failure();
}

// Helpers

/// Returns true if the SameArgumentAndResultTypes trait can be used to infer
/// result types of the given operation.
bool hasSameArgumentAndResultTypes(const Operator& op) {
  return op.getTrait("::mlir::OpTrait::SameOperandsAndResultType") &&
         op.getNumVariableLengthResults() == 0;
}

/// Returns true if the FirstAttrDerivedResultType trait can be used to infer
/// result types of the given operation.
/// TODO: Once the use of this is understood, it should be added to tablegen to
/// simplify builders for ops that use this trait.
// bool hasFirstAttrDerivedResultTypes(const Operator &op) {
//   return op.getTrait("::mlir::OpTrait::FirstAttrDerivedResultType") &&
//          op.getNumVariableLengthResults() == 0;
// }

/// Returns true if the InferTypeOpInterface can be used to infer result types
/// of the given operation.
bool hasInferTypeInterface(const Operator& op) {
  return op.getTrait("::mlir::InferTypeOpInterface::Trait");
}

/// Returns true if there is a trait or interface that can be used to infer
/// result types of the given operation.
bool canInferType(const Operator& op) {
  // TODO: Support hasFirstAttrDerivedResultTypes(op)
  bool hasOutputType = op.getNumResults() > 0;
  return !hasOutputType || hasSameArgumentAndResultTypes(op) ||
         hasInferTypeInterface(op);
}

bool hasVariadicResult(const Operator& op) {
  return llvm::any_of(op.getResults(),
                      [](const auto& result) { return result.isVariadic(); });
}

bool hasSingleVariadicResult(const Operator& op) {
  return op.getNumResults() == 1 && hasVariadicResult(op);
}

bool hasRegions(const Operator& op,
                std::optional<int> numRegions = std::nullopt) {
  if (numRegions.has_value()) {
    return op.getNumRegions() == static_cast<unsigned int>(numRegions.value());
  }
  return op.getNumRegions() > 0;
}

bool isTerminator(const Operator& op) {
  return op.getTrait("::mlir::OpTrait::IsTerminator");
}

// Returns true if we can use unwrapped value for the given `attr` in builders.
bool canUseUnwrappedRawValue(const tblgen::Attribute& attr) {
  return attr.getReturnType() != attr.getStorageType() &&
         // We need to wrap the raw value into an attribute in the builder impl
         // so we need to make sure that the attribute specifies how to do that.
         !attr.getConstBuilderTemplate().empty();
}

///////
// Build Signature
///////

class OpBuilderEmitter {
 public:
  explicit OpBuilderEmitter(const Operator& op) : op_(op) {}

  const Operator& getOp() { return op_; }

  // Return the name of the builder method, the op name without "Op" suffix.
  // I.e. AddOp --> Add
  StringRef getMethodName() {
    // Verify trait
    auto opName = op_.getCppClassName();
    if (opName.ends_with("Op")) return opName.drop_back(2);
    return opName;
  }

  // Return the return type string.
  std::string getReturnType();

  // Get Operand parameters.
  SmallVector<MethodParameter> getOperandParameters();

  SmallVector<MethodParameter> getAttributeParameters();

  // For each region, add a `RegionBuilderCallback` arg
  SmallVector<MethodParameter> getRegionParameters();

  struct BuilderParams {
    // Only set if one of the operands can be used for builder ref
    std::optional<MethodParameter> builderRefOperand;
    // Only set if op does not support type inference, requires explicit type
    std::optional<MethodParameter> outputShape;
    SmallVector<MethodParameter> operands;
    SmallVector<MethodParameter> attributes;
    SmallVector<MethodParameter> regionBuilders;
  };

  // Returns a builder reference from an MlirOp operand, if one exists.
  // If no required operand (not optional or variadic) exists, returns
  // std::nullopt.
  std::optional<MethodParameter> getBuilderFromOperands() {
    auto builderIt = llvm::find_if(op_.getOperands(), [](const auto& operand) {
      return !operand.isOptional() && !operand.isVariadic();
    });
    if (builderIt != op_.getOperands().end()) {
      return MethodParameter("MlirBuilder &",
                             (builderIt->name + ".getBuilder()").str());
    }
    return std::nullopt;
  }

  // Return a default builder reference, if no required operands exist, this
  // parameter must be injected as the first argument.
  MethodParameter getDefaultBuilder() {
    if (isTerminator(getOp())) {
      return MethodParameter("RegionBuilder &", "builder");
    }
    return MethodParameter("MlirBuilder &", "builder");
  }

  // If the op does not support type inference, return a default output shape
  // parameter that must be injected.
  MethodParameter getDefaultOutputShape() {
    return MethodParameter("Type", "resultType");
  }

  // Returns a builder reference from an MlirOp operand, if one exists.
  // If no required operand (not optional or variadic) exists, returns
  // the default builder `builder`.
  MethodParameter getBuilderRef() {
    // Terminators use RegionBuilders.
    if (isTerminator(getOp())) return getDefaultBuilder();

    std::optional<MethodParameter> builderRef = getBuilderFromOperands();
    if (builderRef.has_value()) return std::move(builderRef.value());
    return getDefaultBuilder();
  }

  // Returns a reference to the mlir::OpBuilder, either using MlirOp operand or
  // builder parameter. This is used to create attributes / types.
  MethodParameter getOpBuilderRef() {
    MethodParameter builderRef = getBuilderRef();
    StringRef builderRefName = builderRef.getName();
    return MethodParameter("::mlir::OpBuilder &",
                           (builderRefName + ".getOpBuilder()").str());
  }

  // Returns all parameters needed for builder decl and defs.
  // The ordering and use of these parameters depends on whether this is a decl
  // or def.
  BuilderParams getOpBuilderParameters() {
    BuilderParams params;
    params.builderRefOperand = getBuilderFromOperands();
    if (!canInferType(getOp())) {
      params.outputShape = getDefaultOutputShape();
    }
    params.operands = getOperandParameters();
    params.attributes = getAttributeParameters();
    params.regionBuilders = getRegionParameters();
    return params;
  }

  // Using parameters from getOpBuilderParameters, return a method signature
  // to be used for the builder method decl.
  Method getMethodDecl() {
    // Make a copy to move into method signature.
    BuilderParams params = getOpBuilderParameters();
    SmallVector<MethodParameter> parameters;
    if (!params.builderRefOperand.has_value() || isTerminator(getOp())) {
      parameters.push_back(getDefaultBuilder());
    }
    if (params.outputShape.has_value()) {
      parameters.push_back(params.outputShape.value());
    }
    parameters.append(params.operands.begin(), params.operands.end());

    // Insert RegionBuilders before attributes, since attrs can be optional.
    parameters.append(params.regionBuilders.begin(),
                      params.regionBuilders.end());

    // Push optional / default attributes to the end.
    llvm::sort(params.attributes,
               [](const MethodParameter& a, const MethodParameter& b) {
                 return a.hasDefaultValue() < b.hasDefaultValue();
               });

    parameters.append(params.attributes.begin(), params.attributes.end());
    return Method(getReturnType(), getMethodName(), Method::None,
                  std::move(parameters));
  }

  SmallVector<MethodParameter> getParametersForCall() {
    // Make a copy to move into method signature.
    BuilderParams params = getOpBuilderParameters();
    SmallVector<MethodParameter> parameters;
    if (params.outputShape.has_value()) {
      parameters.push_back(getDefaultOutputShape());
    }
    for (auto& operand : params.operands) {
      parameters.push_back(
          MethodParameter("MlirOp &", "unwrap(" + operand.getName() + ")"));
    }
    // Skip regions, not used in builder calls.
    parameters.append(params.attributes);
    return parameters;
  }

  // Inset a call to the builder method into the given body.
  // I.e. `lhs.getBuilder().create<AddOp>(unwrap(lhs), unwrap(rhs));`
  void buildMethodBody(Method& method);

  // Inset a call to the builder method into the given body.
  // I.e. `lhs.getBuilder().create<AddOp>(unwrap(lhs), unwrap(rhs));`
  void buildMethodCall(MethodBody& body);

  // Insert a creation call and invoke region callbacks.
  void buildMethodCallWithRegions(MethodBody& body);

  // Write a description of the current builder method either to a code comment
  // or a markdown doc string.
  void buildMethodDescription(mlir::raw_indented_ostream& os,
                              StringRef linePrefix);

  // Write a doc string for the current builder method.
  void buildMethodDoc(mlir::raw_indented_ostream& os, Method& method);

 private:
  const Operator& op_;
};

std::string resultsStringSwitch(const Operator& op,
                                std::function<std::string()> zero,
                                std::function<std::string()> one,
                                std::function<std::string(int)> many,
                                std::function<std::string()> variadic) {
  auto numResults = op.getNumResults();
  if (numResults == 0) return zero();
  if (hasSingleVariadicResult(op)) return variadic();
  if (numResults == 1) return one();
  if (numResults > 1) return many(numResults);
  return "<<ResultStringSwitch error>>";
}

// Returns the return type of the builder method.
//   Zero results     --> void
//   One result       --> MlirOp
//   N results        --> SmallVector<MlirOp, N>
//   Single Variadic  --> SmallVector<MlirOp>
std::string OpBuilderEmitter::getReturnType() {
  return resultsStringSwitch(
      getOp(),                    //
      []() { return "void"; },    // zero
      []() { return "MlirOp"; },  // one
      [](int n) { return llvm::formatv("SmallVector<MlirOp, {0}>", n).str(); },
      []() { return "SmallVector<MlirOp>"; });  // variadic
}

// Get operand params:
//   Operand -> MlirOp &
//   Optional Operand -> std::optional<MlirOp>
//   Variadic Operand -> ArrayRef<MlirOp>
SmallVector<MethodParameter> OpBuilderEmitter::getOperandParameters() {
  auto op = getOp();
  SmallVector<MethodParameter> parameters;
  for (const auto& operand : op.getOperands()) {
    if (operand.isOptional()) {
      parameters.emplace_back("std::optional<MlirOp>", operand.name,
                              /*optional=*/true);
      continue;
    }
    if (operand.isVariadic()) {
      parameters.emplace_back("ArrayRef<MlirOp>", operand.name);
      continue;
    }
    // Regular operand.
    parameters.emplace_back("MlirOp &", operand.name);
  }
  return parameters;
}

StringRef getAttributeType(Attribute attr) {
  if (canUseUnwrappedRawValue(attr)) {
    return attr.getReturnType();
  }
  return attr.getStorageType();
}

// Return a default value for an attribute.
//   Optional & Default -> Default
//   Optional & No Default -> {}
//   Default -> Default
std::optional<std::string> getAttributeDefaultValue(OpBuilderEmitter& emitter,
                                                    Attribute attr) {
  if (!attr.isOptional() && !attr.hasDefaultValue()) return std::nullopt;

  FmtContext fctx;
  fctx.withBuilder(emitter.getOpBuilderRef().getName());

  if (canUseUnwrappedRawValue(attr) && attr.hasDefaultValue())
    return tgfmt(attr.getDefaultValue(), &fctx);
  return "{}";
}

// Get attribute params:
// TODO: Support buildable attributes from default values with fmt gen.
SmallVector<MethodParameter> OpBuilderEmitter::getAttributeParameters() {
  auto op = getOp();
  SmallVector<MethodParameter> attributeParameters;
  for (auto& namedAttr : op.getAttributes()) {
    Attribute attr = namedAttr.attr;
    StringRef attrType = getAttributeType(attr);
    std::optional<std::string> defaultValue =
        getAttributeDefaultValue(*this, attr);

    attributeParameters.emplace_back(
        attrType, namedAttr.name, defaultValue.value_or(""), attr.isOptional());
  }
  return attributeParameters;
}

SmallVector<MethodParameter> OpBuilderEmitter::getRegionParameters() {
  SmallVector<MethodParameter> regionParameters;
  for (auto& region : getOp().getRegions()) {
    regionParameters.emplace_back("const RegionBuilderCallback &", region.name);
  }
  return regionParameters;
}

///////
// Build Impl
///////

class ScopedIndent {
 public:
  explicit ScopedIndent(MethodBody& body) : body_(body) { body_.indent(); }
  ~ScopedIndent() { body_.unindent(); }

 private:
  MethodBody& body_;
};

// Zero:     builder.create0<Op>(...)
// One:      builder.create<Op>(...)
// Many:     builder.createN<Op, N>(...)
// Variadic: builder.createVariadic<Op>(...)
void OpBuilderEmitter::buildMethodCall(MethodBody& body) {
  const Operator& op = getOp();
  std::string builderRef = getBuilderRef().getName().str();

  // Build comma separated list of parameters for the call.
  std::string callParams;
  llvm::raw_string_ostream os(callParams);
  llvm::interleaveComma(getParametersForCall(), os,
                        [&](MethodParameter arg) { os << arg.getName(); });

  auto getCallTo = [&](StringRef methodName,
                       std::optional<std::string> n = std::nullopt) {
    // builder.createOp<Op[, N]>(...)
    auto callFmt = "{0}.{1}<{2}{3}>({4});\n";
    return llvm::formatv(callFmt, builderRef, methodName, op.getCppClassName(),
                         n.value_or(""), callParams)
        .str();
  };

  // builder.createUnwrapped<Op>(...)
  if (hasRegions(op)) {
    body << getCallTo("createUnwrapped");
    return;
  }

  body << resultsStringSwitch(
      op, [&]() { return getCallTo("create0"); },  // zero
      [&]() { return getCallTo("create"); },       // one
      [&](int n) { return getCallTo("createN", ", " + std::to_string(n)); },
      [&]() { return getCallTo("createVariadic"); });  // variadic
}

void OpBuilderEmitter::buildMethodCallWithRegions(MethodBody& body) {
  std::string builderRef = getBuilderRef().getName().str();

  // OpTy op = builder.createUnwrapped(...);
  Twine opVar = "_op";
  body << getOp().getCppClassName() << " " << opVar << " = ";
  buildMethodCall(body);

  // RegionBuilder condBuilder(this, &_op->getRegion(1));
  // cond(condBuilder);
  //   {0} = region-name, {1} = builderRef {2} = op-name, {3} = region-idx
  auto buildRegionFmt = R"(
    RegionBuilder _{0}Builder({1}, {2}->getRegion({3}));
    {0}(_{0}Builder);
  )";

  for (auto [idx, region] : llvm::enumerate(getOp().getRegions())) {
    auto impl =
        llvm::formatv(buildRegionFmt, region.name, builderRef, opVar, idx)
            .str();
    body.getStream().printReindented(impl);
  }

  body << "return"
       << resultsStringSwitch(
              getOp(), [&]() { return ";\n"; },  // zero
              [&]() {                            // one
                return llvm::formatv(" MlirOp({0}, {1});\n", builderRef, opVar);
              },
              [&](int n) {  // many
                return llvm::formatv(" wrap({0}, {1}->getResults());\n",
                                     builderRef, opVar)
                    .str();
              },
              [&]() {  // variadic
                return llvm::formatv(" wrap({0}, {1}->getResults());\n",
                                     builderRef, opVar);
              });
}

void OpBuilderEmitter::buildMethodBody(Method& method) {
  MethodBody& body = method.body();
  ScopedIndent indent(body);
  if (hasRegions(getOp())) {
    buildMethodCallWithRegions(body);
    return;
  }
  body << "return ";
  buildMethodCall(body);
}

void OpBuilderEmitter::buildMethodDescription(mlir::raw_indented_ostream& os,
                                              StringRef linePrefix) {
  std::string description;
  llvm::raw_string_ostream ds(description);

  if (isTerminator(op_)) {
    ds << R"(
This operation is a Region's Terminator. It can only be called in a RegionBuilder
function callback when constructing the body of an op.)";
  }
  if (hasRegions(op_)) {
    ds << R"(
This operation has a body region built via a callback function.)";
  }

  if (!description.empty()) {
    os << "\n";
    ds << "\n";
    os.printReindented(description, linePrefix);
  }
}

// Returns a string that is either a link to the spec or the op name.
// The spec link is only generated for ops in dialects that have a spec.
//   known_dialect.op   --> [`known_dialect.op`](spec_link#op)
//   unknown_dialect.op --> `unknown_dialect.op`
std::string maybeSpecLinkedOpName(Operator const& op) {
  std::string opName = op.getOperationName();
  // The format will be filled with the lowercase op name with dialect
  // stripped.
  // TODO: These links dont always work, the latter arg should be the cpp
  // namespaced op in all lowercase.
  DenseMap<StringRef, StringRef> dialectToSpecFmt = {
      // clang-format off
    {"chlo", "https://openxla.org/stablehlo/generated/chlo#chlo{1}_chlo{1}op"},
    {"func", "https://mlir.llvm.org/docs/Dialects/Func/#func{1}-func{1}op"},
    {"sdy", "https://openxla.org/shardy/sdy_dialect#sdy{1}_sdy{1}op"},
    {"stablehlo", "https://openxla.org/stablehlo/spec#{1}"},
    {"tosa", "https://mlir.llvm.org/docs/Dialects/TOSA/#tosa{1}-mlirtosa{1}op"},
      // clang-format on
  };
  auto dialect = op.getDialect().getName();
  if (dialectToSpecFmt.contains(dialect)) {
    StringRef baseUrlFmt = dialectToSpecFmt[dialect];
    StringRef opHref = opName;
    if (opHref.starts_with(dialect)) {
      opHref = opHref.drop_front(dialect.size() + 1);
    }
    std::string urlFmt = "[`{0}`](" + baseUrlFmt.str() + ")";
    return llvm::formatv(urlFmt.c_str(), opName, opHref).str() + "\n";
  }

  return llvm::formatv("`{0}`", opName).str() + " ";
}

void OpBuilderEmitter::buildMethodDoc(mlir::raw_indented_ostream& os,
                                      Method& method) {
  const Operator& op = getOp();

  os << "### `" << op.getDialectName() << "::" << op.getCppClassName() << "`\n";
  os << "\n";
  os << "Creates a new " << maybeSpecLinkedOpName(op) << "operation.\n";
  buildMethodDescription(os, "");
  os << "\n";
  os << "```c++\n";
  method.writeDeclTo(os);
  os << "```\n\n";
}

///////
// Main entry point & validation
///////

LogicalResult verifyReturnType(const Operator& op) {
  bool hasVariadicResult = llvm::any_of(
      op.getResults(), [](const auto& result) { return result.isVariadic(); });
  if (hasVariadicResult && op.getNumResults() > 1)
    return skipOperation(op, "Only single variadic result supported");
  return success();
}

// Must be operands followed by attributes.
LogicalResult verifyArgumentOrder(const Operator& op) {
  bool sawAttr = false;
  for (const auto& arg : op.getArgs()) {
    if (isa<NamedAttribute*>(arg)) {
      sawAttr = true;
      continue;
    }
    if (sawAttr) return skipOperation(op, "Attributes must be after operands");
  }
  if (llvm::any_of(op.getOperands(),
                   [](auto operand) { return operand.isOptional(); }))
    return skipOperation(op, "Optional operands not supported.");
  return success();
}

LogicalResult verifyAttributes(const Operator& op) {
  // TODO: Name conflicts cause issues, like StableHLO Transpose attr vs
  // the free stablehlo::Transpose op builder method. The StableHLO enum kind
  // should be renamed.
  llvm::DenseSet<StringRef> knownBadTypes = {"StableHLO_TransposeAttr"};
  bool hasBadType =
      llvm::any_of(op.getAttributes(), [&](const NamedAttribute& attr) {
        return knownBadTypes.contains(attr.attr.getDefName());
      });
  if (hasBadType) return skipOperation(op, "Attributes have known bad types");
  return success();
}

LogicalResult verifyRegions(const Operator& op) {
  if (llvm::any_of(op.getRegions(),
                   [](const auto& region) { return region.isVariadic(); }))
    return skipOperation(op, "Variadic regions not supported");
  return success();
}

LogicalResult verifyOpTraits(const Operator& op) {
  if (op.getTrait("::mlir::FunctionOpInterface::Trait"))
    return skipOperation(op, "FunctionOpInterface not supported");
  if (op.skipDefaultBuilders())
    return skipOperation(op, "Op does not use MLIR's default builders");
  return success();
}

// Returns an OpBuilderEmitter if possible.
// This is mostly limited by features, and more ops can have op builder emitters
// as feature support is added.
// Some supported patterns:
// - [X] Op has one or more Value operands.
// - [X] Op has one or more results.
// - [X] Op has no required attributes.
// - [X] Op has no operands.
// - [X] Op cannot infer type (take result type as argument).
// - [X] Op has no results.
// - [X] Op has no required MlirOp operands.
// - [X] Op has single variadic operand / result.
// - [X] Op has required attributes.
// - [X] Op has optional attribute followed by non-optional attribute.
// - [ ] Op has a region.
// - [ ] Op has multiple operands results, with some variadic(s).
// - [ ] Op Optional operands.
// - [ ] Op declares attributes before operands (chlo.constant_like).
// - [ ] Op uses `FirstAttrDerivedResultTypes` to infer result type (tosa.const)
// - [ ] Op method is a name conflict (triangular_solve Transpose is enum & fn).
// - [ ] Op does not use MLIR's default builders.
FailureOr<OpBuilderEmitter> getAndVerifyOpBuilderEmitter(const Operator& op) {
  // Verify return type
  if (failed(verifyReturnType(op))) return failure();

  // Verify arg order is operands -> attributes
  if (failed(verifyArgumentOrder(op))) return failure();

  // Verify attributes
  if (failed(verifyAttributes(op))) return failure();

  // Verify Regions -- no variadic regions yet.
  if (failed(verifyRegions(op))) return failure();

  // Verify op traits -- no FunctionOpInterface yet.
  if (failed(verifyOpTraits(op))) return failure();

  return OpBuilderEmitter(op);
}

void WriteOperatorBuilder(OpBuilderEmitter& emitter,
                          mlir::raw_indented_ostream& os) {
  Method method = emitter.getMethodDecl();
  emitter.buildMethodBody(method);

  // ASSUMPTION: Operation has at least one operand.
  // Need to switch on operations that don't have operands to take a builder as
  // an argument.
  switch (action) {
    case ActionType::GenBuilderHeader:
      method.writeDeclTo(os);
      return;
    case ActionType::GenBuilderImpl:
      method.writeDefTo(os, "");
      return;
    case ActionType::GenBuilderDocs:
      emitter.buildMethodDoc(os, method);
      return;
  }
  llvm::report_fatal_error("[WriteOperatorBuilder] Unknown enum value.");
}

void writeFileHeader(mlir::raw_indented_ostream& os, StringRef header) {
  if (action == ActionType::GenBuilderDocs) {
    os << "# " << header << "\n\n";
    os << "[TOC]\n\n";
    os << "## Builder Methods\n\n";
    return;
  }
  emitSourceFileHeader(header, os);
}

void writeSkipped(mlir::raw_indented_ostream& os,
                  std::vector<Operator> skipped) {
  if (skipped.empty()) return;

  std::string prefix = "// Skipped ";
  if (action == ActionType::GenBuilderDocs) {
    prefix = " - ";
    os << "## Skipped Operations\n\n";
    os << "Unable to generate builder for the following operations:\n\n";
  }
  for (const auto& op : skipped) {
    os << prefix << maybeSpecLinkedOpName(op) << "\n";
  }
}

// The function below has a non-constant reference as that is required by LLVM's
// TableGenMain.
// NOLINTNEXTLINE
bool GenerateStablehloBuilderMain(raw_ostream& os,
                                  const RecordKeeper& records) {
  mlir::raw_indented_ostream indentedOs(os);
  // Get the list of StableHLO operations that are allowed to be directly
  // converted to HLO without intermediate MHLO step.

  // Emit file header.
  auto opList = records.getAllDerivedDefinitions("Op");
  auto dialect = Operator(opList[0]).getDialect().getName();
  auto header = ("`" + dialect + "` MLIR Dialect Builder API").str();
  writeFileHeader(indentedOs, header);

  // Emit all the MLIR Builders
  std::vector<Operator> skipped;
  for (const auto* def : records.getAllDerivedDefinitions("Op")) {
    Operator op(def);
    FailureOr<OpBuilderEmitter> emitter = getAndVerifyOpBuilderEmitter(op);
    if (failed(emitter)) {
      skipped.push_back(op);
      continue;
    }
    WriteOperatorBuilder(emitter.value(), indentedOs);
  }

  writeSkipped(indentedOs, skipped);

  return false;
}

}  // namespace
}  // namespace mlir

int main(int argc, char** argv) {
  llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);
  llvm::InitLLVM y(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv);
  return TableGenMain(argv[0], &mlir::GenerateStablehloBuilderMain);
}
