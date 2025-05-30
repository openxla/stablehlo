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

#include "stablehlo/transforms/StablehloRefineShapes.h"

#include <cstddef>
#include <cstdint>
#include <tuple>
#include <utility>

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/Base.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/dialect/TypeInference.h"
#include "stablehlo/transforms/Passes.h"
#include "stablehlo/transforms/optimization/Passes.h"

#define DEBUG_TYPE "stablehlo-refine-shapes"

namespace mlir {
namespace stablehlo {

#define GEN_PASS_DEF_STABLEHLOREFINESHAPESPASS
#include "stablehlo/transforms/Passes.h.inc"

LogicalResult refineValues(PatternRewriter& rewriter, Operation* op,
                           ValueRange values, TypeRange types) {
  if (values.size() != types.size())
    return rewriter.notifyMatchFailure(op, [&](Diagnostic& diag) {
      diag << "refineValues failed for " << types << ": expected "
           << values.size() << " types, got " << types.size();
    });

  // Check whether `types` contain any new information with respect to
  // existing return types. Even if just a single dimension size out of an
  // entire tensor type got updated, using `inferMostSpecificType` ensures
  // that we don't miss that.
  bool needsRefinement = false;
  SmallVector<Type> refinedTypes;
  for (auto it : llvm::zip(values.getTypes(), types)) {
    // Cannot use structured bindings to simplify this because capturing
    // structured bindings in a lambda is a C++ 20 extension.
    auto currentType = std::get<0>(it);
    auto refinement = std::get<1>(it);
    auto refinedType = hlo::inferMostSpecificType(
        /*location=*/{}, {currentType, refinement});
    if (failed(refinedType)) {
      return rewriter.notifyMatchFailure(op, [&](Diagnostic& diag) {
        diag << "inferMostSpecificType failed for " << currentType << " and "
             << refinement;
      });
    }
    refinedTypes.push_back(*refinedType);
    needsRefinement |= (currentType != *refinedType);
  }
  if (!needsRefinement)
    return rewriter.notifyMatchFailure(op, "doesn't need refinement");

  for (auto it : llvm::zip(values, refinedTypes)) {
    // Cannot use structured bindings to simplify this because capturing
    // structured bindings in a lambda is a C++ 20 extension.
    auto value = std::get<0>(it);
    auto refinedType = std::get<1>(it);
    if (value.getType() == refinedType) continue;

    // Check whether the users of this value are ready for the type of the
    // value to be refined.
    for (Operation* user : value.getUsers()) {
      // CHLO and StableHLO ops are designed to support type refinements of
      // their operands and results. Any operand type in these ops can change
      // within what's supported by `inferMostSpecificType` without breaking
      // verification of the op.
      if (isa<chlo::ChloDialect, StablehloDialect>(user->getDialect()))
        continue;
      // TODO(bartchr): Consider if the dialect allow-listing approach is too
      // strict. In the meantime, allow some shape interop with the shardy
      // dialect.
      if (user->getDialect()->getNamespace() == "sdy") continue;

      // Simply changing operand type of `func.return` won't work because
      // that won't update the FunctionType of the enclosing `func.func`.
      if (isa<func::ReturnOp>(user)) continue;
      if (isa<func::CallOp>(user)) continue;

      // Unlike in TensorFlow's type inference pass, here we work only with
      // allowlisted ops to focus our support on well-defined semantics of
      // StableHLO programs.
      return rewriter.notifyMatchFailure(op, [&](Diagnostic& diag) {
        diag << "unsupported refinement: tried to refine " << value.getType()
             << " to " << refinedType << " for user " << user;
      });
    }

    // Happy path: simply call setType here because most of our users are
    // fine with that.
    auto unrefinedType = value.getType();
    value.setType(refinedType);

    // Special case: for `func.return`, guard the refinement with a cast
    // and leave propagation of the refined return type to a dedicated pattern.
    auto isFuncReturn = [](OpOperand& use) -> bool {
      return isa<func::ReturnOp>(use.getOwner());
    };
    if (llvm::none_of(value.getUses(), isFuncReturn)) continue;
    rewriter.setInsertionPointAfter(op);
    auto castToUnrefinedType = rewriter.create<UnrealizedConversionCastOp>(
        op->getLoc(), unrefinedType, value);
    value.replaceUsesWithIf(castToUnrefinedType.getOutputs()[0], isFuncReturn);
  }

  return success();
}

LogicalResult refineReturnTypes(PatternRewriter& rewriter, Operation* op,
                                ArrayRef<Type> types) {
  if (failed(refineValues(rewriter, op, op->getResults(), types)))
    return failure();

  // This `replaceOpUsesWithIf` call doesn't actually change the IR, but
  // it does ask the rewriter to visit all the users of this op. There is no
  // upstream API to achieve this directly, but if it's introduced in the
  // future, we could use it here.
  rewriter.replaceOpUsesWithIf(op, op->getResults(),
                               [](OpOperand& use) { return false; });
  return success();
}

LogicalResult refineReturnTypes(PatternRewriter& rewriter, Operation* op,
                                ArrayRef<ShapedTypeComponents> refinements) {
  SmallVector<Type> flattenedTypes;
  hlo::flattenTupleTypes(op->getResultTypes(), flattenedTypes);
  auto flattenedSize = flattenedTypes.size();
  if (flattenedSize != refinements.size())
    return rewriter.notifyMatchFailure(op, [&](Diagnostic& diag) {
      diag << "refineReturnTypes failed: expected " << flattenedSize
           << " refinements, got " << refinements.size();
    });

  SmallVector<Type> flattenedRefinedTypes;
  for (auto it : llvm::zip(flattenedTypes, refinements)) {
    // Cannot use structured bindings to simplify this because capturing
    // structured bindings in a lambda is a C++ 20 extension.
    ShapedType currentType = dyn_cast<ShapedType>(std::get<0>(it));
    ShapedTypeComponents refinement = std::get<1>(it);
    auto failWithReason = [&](StringRef reason) {
      return rewriter.notifyMatchFailure(op, [&](Diagnostic& diag) {
        diag << "refineTypes failed: refining " << currentType
             << "with refinement: {";
        if (refinement.hasRank()) {
          diag << "shape = [" << refinement.getDims() << "]";
          if (refinement.getAttribute())
            diag << "attribute = " << refinement.getAttribute();
        } else {
          diag << "hasRank = false";
        }
        diag << ", elementType = " << refinement.getElementType();
        diag << "} failed: " << reason;
      });
    };

    // If the current type is not a shaped type, then the refinement must
    // be completely empty.
    if (!currentType) {
      if (refinement.hasRank() || refinement.getElementType() ||
          refinement.getAttribute())
        return failWithReason("unsupported refinement");
      flattenedRefinedTypes.push_back(currentType);
      continue;
    }

    // If the refinement has an element type, then it must be the same as
    // the current element type.
    Type currentElementType = currentType.getElementType();
    if (refinement.getElementType() &&
        currentElementType != refinement.getElementType())
      return failWithReason("expected compatible element types");

    // If neither the current type nor the refinement are ranked, then there's
    // nothing to refine, and we return the current type.
    bool hasRank = currentType.hasRank() || refinement.hasRank();
    if (!hasRank) {
      flattenedRefinedTypes.push_back(currentType);
      continue;
    }

    // If either the current type or the refinement have encodings, then
    // we fail. Encodings are left for future work.
    Attribute currentEncoding = nullptr;
    if (auto currentRankedType = dyn_cast<RankedTensorType>(currentType)) {
      currentEncoding = currentRankedType.getEncoding();
    }
    Attribute refinedEncoding = refinement.getAttribute();
    if (currentEncoding || refinedEncoding)
      return failWithReason("expected compatible encodings");

    // If both the current type and the refinement have shapes, use the shape
    // from the refinement. Otherwise, pick whatever is available.
    // Make sure that the resulting type is compatible with the current type
    // to avoid creating invalid code.
    auto refinedShape =
        refinement.hasRank() ? refinement.getDims() : currentType.getShape();
    auto refinedType = RankedTensorType::get(refinedShape, currentElementType);
    if (!hlo::isCompatibleForHloTypeInference(currentType, refinedType))
      return failWithReason("expected compatible shapes");
    flattenedRefinedTypes.push_back(refinedType);
  }

  SmallVector<Type> refinedTypes;
  if (failed(hlo::unflattenTupleTypes(op->getResultTypes(),
                                      flattenedRefinedTypes, refinedTypes)))
    return failure();
  return refineReturnTypes(rewriter, op, refinedTypes);
}

namespace {

class RefinementKey {
 public:
  RefinementKey(func::FuncOp func, int64_t leadingTokenOperands,
                SmallVector<APSInt> const& globalConstants,
                SmallVector<Type> const& functionalArgumentTypes)
      : func(func),
        leadingTokenOperands(leadingTokenOperands),
        globalConstants(globalConstants),
        functionalArgumentTypes(functionalArgumentTypes) {}

  static FailureOr<RefinementKey> fromCallOp(func::CallOp callOp) {
    LLVM_DEBUG(llvm::dbgs() << "RefinementKey::fromCallOp: "
                            << callOp.getCalleeType() << "\n");
    int64_t leadingTokenOperands = countLeadingTokenOperands(callOp);
    SmallVector<APSInt> globalConstants =
        getGlobalConstants(callOp, leadingTokenOperands);
    SmallVector<Type> functionalArgumentTypes = getFunctionalArgumentTypes(
        callOp, leadingTokenOperands, globalConstants.size());

    FlatSymbolRefAttr calleeName = callOp.getCalleeAttr();
    const SymbolTable symbolTable(callOp->getParentOfType<ModuleOp>());
    auto callee = symbolTable.lookupNearestSymbolFrom<func::FuncOp>(
        callOp, calleeName.getAttr());
    if (!callee) return callOp.emitOpError() << "cannot resolve function call";
    return RefinementKey(callee, leadingTokenOperands, globalConstants,
                         functionalArgumentTypes);
  }

  // Getters
  func::FuncOp getFunc() const { return func; }
  int64_t getLeadingTokenOperands() const { return leadingTokenOperands; }
  SmallVector<APSInt> const& getGlobalConstants() const {
    return globalConstants;
  }
  SmallVector<Type> const& getFunctionalArgumentTypes() const {
    return functionalArgumentTypes;
  }

  // Get all non global-constant args, including tokens and functional args.
  SmallVector<Type> getAllNonGlobalConstantArgumentTypes(
      MLIRContext& context) const {
    SmallVector<Type> types(getLeadingTokenOperands() +
                            getFunctionalArgumentTypes().size());
    for (size_t i = 0; i < static_cast<size_t>(leadingTokenOperands); ++i)
      types[i] = stablehlo::TokenType::get(&context);
    for (auto [i, refinedType] : llvm::enumerate(getFunctionalArgumentTypes()))
      types[i + leadingTokenOperands] = refinedType;
    return types;
  }

  // Utilities
  inline std::string toString() {
    std::string buffer;
    llvm::raw_string_ostream os(buffer);
    os << "RefinementKey(" << func.getName()
       << ", toks=" << leadingTokenOperands << ", dim_args=[";
    llvm::interleaveComma(globalConstants, os);
    os << "], fn_args=[";
    llvm::interleaveComma(functionalArgumentTypes, os);
    os << "])";
    return buffer;
  }

 private:
  static int64_t countLeadingTokenOperands(func::CallOp callOp) {
    int64_t nrLeadingTokenOperands = 0;
    for (auto operand : callOp.getOperands()) {
      if (!isa<TokenType>(operand.getType())) break;
      nrLeadingTokenOperands++;
    }
    return nrLeadingTokenOperands;
  }

  // global-constant arguments follow token args, and are scalar integer
  // constants These represent the known values of symbolic shapes sizes. I.e.
  // tensor<Axf32> : A = constant(5)
  static SmallVector<APSInt> getGlobalConstants(func::CallOp callOp,
                                                int64_t leadingTokenOperands) {
    SmallVector<APSInt> globalConstants;
    auto operands = callOp.getOperands();
    for (size_t i = leadingTokenOperands; i < operands.size(); ++i) {
      auto operandType = dyn_cast<RankedTensorType>(operands[i].getType());
      if (!operandType || operandType.getRank() != 0 ||
          !operandType.getElementType().isInteger())
        break;

      SmallVector<APSInt> operand_int;
      if (failed(hlo::matchInts(operands[i], operand_int))) break;
      globalConstants.push_back(operand_int[0]);
    }
    return globalConstants;
  }

  // Functional operands are the arguments that are not global-constant
  // arguments. These are the values that will remain after symbolic shape
  // refinement.
  static SmallVector<Type> getFunctionalArgumentTypes(
      func::CallOp callOp, int64_t leadingTokenOperands,
      int64_t globalConstantsSize) {
    SmallVector<Type> functionalArgumentTypes;
    auto operands = callOp.getOperands();
    for (size_t i = leadingTokenOperands + globalConstantsSize;
         i < operands.size(); ++i) {
      functionalArgumentTypes.push_back(operands[i].getType());
    }
    return functionalArgumentTypes;
  }

 private:
  func::FuncOp func;
  int64_t leadingTokenOperands;
  SmallVector<APSInt> globalConstants;
  SmallVector<Type> functionalArgumentTypes;
};

// Per-module state for shape refinement.
// An entry is Key is <FuncOp, SmallVector<APSInt>, SmallVector<Type>>
// Which correlates to <func, sym_int_values, arg_types>
class RefineShapeState {
 public:
  RefineShapeState(
      std::optional<AdditionalShapeRefinementPatternsFn> additionalPatternsFn)
      : additionalPatternsFn(additionalPatternsFn) {}

  enum class RefinementState {
    NOT_ALREADY_REFINED,
    ALREADY_REFINED,
  };

  // Validates that we are not attempting to refine a function with a different
  // context than previously, and are not attempting recursive refinement.
  // Returns failure() if validation fails. On success, returns a refinement
  // state that specifies whether the function has already been refined.
  FailureOr<RefinementState> validateFunctionRefinement(RefinementKey key) {
    func::FuncOp func = key.getFunc();
    StringRef funcName = func.getName();

    auto found = refinementContexts.find(func);
    if (found == refinementContexts.end())
      return RefinementState::NOT_ALREADY_REFINED;
    RefinementKey prevKey = found->second;

    // Since we refine until fixed point, we will refine a call to a function
    // both for the original function and for the refined one. In the latter
    // case, we should have empty globalConstants but everything else the
    // same.
    if (!key.getGlobalConstants().empty() &&
        prevKey.getGlobalConstants() != key.getGlobalConstants())
      return emitDifferentRefinementContextError(key.getFunc(), key, prevKey);

    // Check that all non-global-constant arguments are the same.
    // Must compare all non-global-constant types, since tokens may become
    // leading:
    //  Refine iter1: `token, dim, token, arg` : 1 leading token
    //  Refine iter2: `token, token, arg` : 2 leading tokens
    MLIRContext& context = *func.getContext();
    if (key.getAllNonGlobalConstantArgumentTypes(context) !=
        prevKey.getAllNonGlobalConstantArgumentTypes(context))
      return emitDifferentRefinementContextError(key.getFunc(), key, prevKey);

    // Don't allow recursive refinement.
    if (llvm::is_contained(functionsBeingRefined, funcName))
      return func.emitOpError()
             << "Function " << funcName << " is being refined recursively\n";

    return RefinementState::ALREADY_REFINED;
  }

  // Updates the state to signal the starting of a function refinement.
  // Callers must call `finishFunctionRefinement` when done.
  [[nodiscard]] auto createScopedFunctionRefinement(RefinementKey& key) {
    func::FuncOp func = key.getFunc();
    auto funcName = func.getName();
    functionsBeingRefined.push_back(funcName);
    refinementContexts.try_emplace(func, key);
    // Return a cleanup function that will pop the function from the stack
    // when it goes out of scope. This can only use values that will have the
    // same lifetime as cleanup fn. In this case, `this` and `key` are safe.
    return llvm::make_scope_exit([this, &key]() {
      if (key.getFunc().getName() != functionsBeingRefined.back())
        llvm::report_fatal_error(
            "Stack mismatch in createScopedFunctionRefinement");
      functionsBeingRefined.pop_back();
    });
  }

  void addAdditionalPatterns(RewritePatternSet& patterns) {
    if (additionalPatternsFn.has_value())
      additionalPatternsFn.value()(&patterns);
  }

 private:
  std::optional<AdditionalShapeRefinementPatternsFn> additionalPatternsFn;

  // Maps refined functions to the refinement context: the values of dimension
  // arguments and the types of non-global-constant arguments. A function is
  // added here when we start refining it.
  DenseMap<func::FuncOp, RefinementKey> refinementContexts;

  // A stack of functions that are in the process of being refined, the current
  // one is last.
  SmallVector<llvm::StringRef> functionsBeingRefined;

  LogicalResult emitDifferentRefinementContextError(func::FuncOp func,
                                                    RefinementKey key,
                                                    RefinementKey prevKey) {
    return func.emitOpError() << "refined with invompatible refinement keys:"
                              << "\n  curr=" << key.toString()
                              << "\n  prev=" << prevKey.toString();
  }
};

// Forward declaration
LogicalResult refineFunction(MLIRContext& context, RefineShapeState& state,
                             RefinementKey& key);

// Check if a function only returns constant values, if so, return the constant
// values that it returns.
std::optional<SmallVector<DenseIntElementsAttr>> isConstantFunction(
    func::FuncOp func) {
  LLVM_DEBUG(llvm::dbgs() << "check if " << func.getName()
                          << " is a constant function\n");
  SmallVector<DenseIntElementsAttr> returnedConstants;
  func::ReturnOp ret = *func.getOps<func::ReturnOp>().begin();
  bool isConstant = llvm::all_of(ret->getOperands(), [&](auto returnVal) {
    DenseIntElementsAttr attr;
    Operation* return_operand_def = returnVal.getDefiningOp();
    if (return_operand_def &&
        matchPattern(return_operand_def, m_Constant(&attr))) {
      returnedConstants.push_back(attr);
      return true;
    }
    return false;
  });
  if (isConstant) return returnedConstants;
  return std::nullopt;
}

// The patterns below implement shape refinement of individual ops.
// In a nutshell, they use the upstream type inference infrastructure and a
// StableHLO-specific extension to refine return types based on potentially
// refined operands.

struct RefineAllGatherOpPattern : public OpRewritePattern<AllGatherOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(AllGatherOp op,
                                PatternRewriter& rewriter) const override {
    for (auto operand : op->getOperands()) {
      auto operandType = cast<ShapedType>(operand.getType());

      // This represents the cross_replica_and_partition process grouping
      // strategy that requires num_partitions to compute shardCount. Since we
      // don't know num_partitions at this point, we error out.
      if (op.getChannelHandle() && !op.getUseGlobalDeviceIds())
        return rewriter.notifyMatchFailure(op, "unsupported strategy");
      DenseIntElementsAttr replicaGroups = op.getReplicaGroups();
      auto shardCount = replicaGroups.getType().getDimSize(1);
      SmallVector<int64_t> refinement(operandType.getShape());
      if (!operandType.isDynamicDim(op.getAllGatherDim()))
        refinement[op.getAllGatherDim()] *= shardCount;
      auto status = refineReturnShape(rewriter, op, refinement);
      if (status.failed()) return status;
    }
    return success();
  }
};

struct RefineBitcastConvertOpPattern
    : public OpRewritePattern<BitcastConvertOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(BitcastConvertOp op,
                                PatternRewriter& rewriter) const override {
    auto operandType = op.getOperand().getType();

    // If bit widths of the operand and the result are different, then
    // operand and result shapes have different ranks.
    // This complicates the logic quite a bit and is not needed to pass the
    // current tests, so we leave this for future work.
    auto resultType = op.getType();
    auto getBitWidthFn = [](ShapedType type) {
      auto elementType = type.getElementType();
      if (auto complexType = dyn_cast<ComplexType>(elementType))
        return complexType.getElementType().getIntOrFloatBitWidth();
      return elementType.getIntOrFloatBitWidth();
    };

    if (getBitWidthFn(operandType) != getBitWidthFn(resultType))
      return rewriter.notifyMatchFailure(op, "unsupported bit width");

    return refineReturnShape(rewriter, op, operandType.getShape());
  }
};

struct RefineCallOpPattern : public OpRewritePattern<func::CallOp> {
  using OpRewritePattern::OpRewritePattern;

  RefineCallOpPattern(MLIRContext* context, RefineShapeState& state)
      : OpRewritePattern<func::CallOp>(context), state(state) {}

  LogicalResult matchAndRewrite(func::CallOp op,
                                PatternRewriter& rewriter) const override {
    auto refinementKey = RefinementKey::fromCallOp(op);
    if (failed(refinementKey)) return failure();
    if (failed(refineFunction(*rewriter.getContext(), state, *refinementKey)))
      return failure();

    // Is the callee a constant function in this refinement context?
    auto callee = refinementKey->getFunc();
    std::optional<SmallVector<DenseIntElementsAttr>> constantAttrs =
        isConstantFunction(callee);
    if (constantAttrs.has_value()) {
      SmallVector<Value> constants;
      for (auto constAttr : constantAttrs.value()) {
        constants.push_back(
            rewriter.create<ConstantOp>(op.getLoc(), constAttr));
      }
      rewriter.replaceOp(op, constants);
      return success();
    }
    if (!refinementKey->getGlobalConstants().empty()) {
      // Drop the global-constant arguments, but only if necessary, or else we
      // will end up trying to refine the new CallOp forever.
      SmallVector<Value> newOperands;
      auto leadingTokenOperands =
          op.getOperands().take_front(refinementKey->getLeadingTokenOperands());
      auto functionalOperands = op.getOperands().take_back(
          refinementKey->getFunctionalArgumentTypes().size());
      newOperands.append(leadingTokenOperands.begin(),
                         leadingTokenOperands.end());
      newOperands.append(functionalOperands.begin(), functionalOperands.end());
      op = rewriter.replaceOpWithNewOp<func::CallOp>(
          op, op.getResultTypes(), callee.getSymName(), newOperands);
      LLVM_DEBUG(llvm::dbgs() << "Replaced call with " << op << "\n");
    }
    return refineReturnTypes(rewriter, op, callee.getResultTypes());
  }

 private:
  RefineShapeState& state;
};

struct RefineConvertOpPattern : public OpRewritePattern<ConvertOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ConvertOp op,
                                PatternRewriter& rewriter) const override {
    SmallVector<ShapedTypeComponents> inferredReturnShapes;
    if (failed(hlo::inferConvertOp(
            /*location=*/{}, op.getOperand(), inferredReturnShapes)))
      return rewriter.notifyMatchFailure(op, "inferConvertOp failed");
    return refineReturnTypes(rewriter, op, inferredReturnShapes);
  }
};

struct RefineConvolutionOpPattern : public OpRewritePattern<ConvolutionOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ConvolutionOp op,
                                PatternRewriter& rewriter) const override {
    SmallVector<ShapedTypeComponents> inferredReturnShapes;
    if (failed(hlo::inferConvolutionOp(
            /*location=*/{}, op.getLhs().getType(), op.getRhs().getType(),
            op.getWindowStrides(), op.getPadding(), op.getLhsDilation(),
            op.getRhsDilation(), op.getWindowReversal(),
            op.getDimensionNumbers().getInputBatchDimension(),
            op.getDimensionNumbers().getInputFeatureDimension(),
            op.getDimensionNumbers().getInputSpatialDimensions(),
            op.getDimensionNumbers().getKernelInputFeatureDimension(),
            op.getDimensionNumbers().getKernelOutputFeatureDimension(),
            op.getDimensionNumbers().getKernelSpatialDimensions(),
            op.getDimensionNumbers().getOutputBatchDimension(),
            op.getDimensionNumbers().getOutputFeatureDimension(),
            op.getDimensionNumbers().getOutputSpatialDimensions(),
            op.getFeatureGroupCount(), op.getBatchGroupCount(),
            op.getPrecisionConfig(), inferredReturnShapes)))
      return rewriter.notifyMatchFailure(op, "inferConvolutionOp failed");
    return refineReturnTypes(rewriter, op, inferredReturnShapes);
  }
};

struct RefineCustomCallOpPattern : public OpRewritePattern<CustomCallOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(CustomCallOp op,
                                PatternRewriter& rewriter) const override {
    SmallVector<ShapedTypeComponents> refinements;
    if (failed(hlo::getShapeRefinements(op.getLoc(), op, refinements)))
      return rewriter.notifyMatchFailure(op, "expected valid refinements");
    if (failed(refineReturnTypes(rewriter, op, refinements)))
      return rewriter.notifyMatchFailure(op, "refineReturnTypes failed");

    // Clean up operand buffers after refinement
    // Must do in this pattern to avoid needing multiple refinement iterations
    if (op.getCallTargetName() == kCustomCallOperandBarrierTarget) {
      Value operand = op.getOperand(0);
      if (operand.getType() == op.getResult(0).getType()) {
        op.replaceAllUsesWith(ValueRange(operand));
      }
      op.erase();
    }
    return success();
  }
};

struct RefineDotGeneralOpPattern : public OpRewritePattern<DotGeneralOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(DotGeneralOp op,
                                PatternRewriter& rewriter) const override {
    SmallVector<ShapedTypeComponents> inferredReturnShapes;
    if (failed(hlo::inferDotGeneralOp(
            /*location=*/{}, op.getLhs().getType(), op.getRhs().getType(),
            op.getDotDimensionNumbersAttr().getLhsBatchingDimensions(),
            op.getDotDimensionNumbersAttr().getRhsBatchingDimensions(),
            op.getDotDimensionNumbersAttr().getLhsContractingDimensions(),
            op.getDotDimensionNumbersAttr().getRhsContractingDimensions(),
            op.getPrecisionConfig(), inferredReturnShapes)))
      return rewriter.notifyMatchFailure(op, "inferDotGeneralOp failed");
    return refineReturnTypes(rewriter, op, inferredReturnShapes);
  }
};

struct RefineDotOpPattern : public OpRewritePattern<DotOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(DotOp op,
                                PatternRewriter& rewriter) const override {
    SmallVector<ShapedTypeComponents> inferredReturnShapes;
    if (failed(hlo::inferDotOp(
            /*location=*/{}, op.getLhs().getType(), op.getRhs().getType(),
            op.getPrecisionConfig(), inferredReturnShapes)))
      return rewriter.notifyMatchFailure(op, "inferDotOp failed");
    return refineReturnTypes(rewriter, op, inferredReturnShapes);
  }
};

struct RefineDynamicBroadcastInDimOpPattern
    : public OpRewritePattern<DynamicBroadcastInDimOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(DynamicBroadcastInDimOp op,
                                PatternRewriter& rewriter) const override {
    return refineReturnShape(rewriter, op, op.getOutputDimensions());
  }
};

struct RefineDynamicConvOpPattern : public OpRewritePattern<DynamicConvOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(DynamicConvOp op,
                                PatternRewriter& rewriter) const override {
    SmallVector<int64_t> padding;
    if (failed(hlo::matchInts(op.getPadding(), padding)))
      return rewriter.notifyMatchFailure(op, "expected constant padding");
    auto paddingType = RankedTensorType::get(
        op.getPadding().getType().getShape(), rewriter.getIntegerType(64));
    auto paddingAttr = DenseIntElementsAttr::get(paddingType, padding);

    SmallVector<ShapedTypeComponents> inferredReturnShapes;
    if (failed(hlo::inferConvolutionOp(
            /*location=*/{}, op.getLhs().getType(), op.getRhs().getType(),
            op.getWindowStrides(), paddingAttr, op.getLhsDilation(),
            op.getRhsDilation(), op.getWindowReversal(),
            op.getDimensionNumbers().getInputBatchDimension(),
            op.getDimensionNumbers().getInputFeatureDimension(),
            op.getDimensionNumbers().getInputSpatialDimensions(),
            op.getDimensionNumbers().getKernelInputFeatureDimension(),
            op.getDimensionNumbers().getKernelOutputFeatureDimension(),
            op.getDimensionNumbers().getKernelSpatialDimensions(),
            op.getDimensionNumbers().getOutputBatchDimension(),
            op.getDimensionNumbers().getOutputFeatureDimension(),
            op.getDimensionNumbers().getOutputSpatialDimensions(),
            op.getFeatureGroupCount(), op.getBatchGroupCount(),
            op.getPrecisionConfig(), inferredReturnShapes)))
      return rewriter.notifyMatchFailure(op, "inferConvolutionOp failed");
    return refineReturnTypes(rewriter, op, inferredReturnShapes);
  }
};

struct RefineDynamicIotaOpPattern : public OpRewritePattern<DynamicIotaOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(DynamicIotaOp op,
                                PatternRewriter& rewriter) const override {
    return refineReturnShape(rewriter, op, op.getOutputShape());
  }
};

struct RefineDynamicPadOpPattern : public OpRewritePattern<DynamicPadOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(DynamicPadOp op,
                                PatternRewriter& rewriter) const override {
    // At the moment, we only support refining return types using fully static
    // shape values which serves the current use cases well.
    // Support for partially static shape values is left for future work.
    SmallVector<int64_t> edgePaddingLow, edgePaddingHigh, interiorPadding;
    if (failed(hlo::matchInts(op.getEdgePaddingLow(), edgePaddingLow)))
      return rewriter.notifyMatchFailure(op,
                                         "expected constant edge_padding_low");
    if (failed(hlo::matchInts(op.getEdgePaddingHigh(), edgePaddingHigh)))
      return rewriter.notifyMatchFailure(op,
                                         "expected constant edge_padding_high");
    if (failed(hlo::matchInts(op.getInteriorPadding(), interiorPadding)))
      return rewriter.notifyMatchFailure(op,
                                         "expected constant interior_padding");

    SmallVector<Type> inferredReturnTypes;
    if (failed(hlo::inferPadOp(
            /*location=*/{}, op.getOperand().getType(),
            op.getPaddingValue().getType(), edgePaddingLow, edgePaddingHigh,
            interiorPadding, inferredReturnTypes)))
      return rewriter.notifyMatchFailure(op, "inferPadOp failed");
    return refineReturnTypes(rewriter, op, inferredReturnTypes);
  }
};

struct RefineDynamicReshapeOpPattern
    : public OpRewritePattern<DynamicReshapeOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(DynamicReshapeOp op,
                                PatternRewriter& rewriter) const override {
    return refineReturnShape(rewriter, op, op.getOutputShape());
  }
};

struct RefineInferTypeOpInterfacePattern
    : public OpInterfaceRewritePattern<InferTypeOpInterface> {
  explicit RefineInferTypeOpInterfacePattern(MLIRContext* context)
      : OpInterfaceRewritePattern(context, /*benefit=*/0) {}
  LogicalResult matchAndRewrite(InferTypeOpInterface op,
                                PatternRewriter& rewriter) const override {
    // Unlike in TensorFlow's type inference pass, here we work only with
    // allowlisted ops to focus our support on well-defined semantics of
    // StableHLO programs.
    if (!isa<chlo::ChloDialect, StablehloDialect>(op->getDialect()))
      return rewriter.notifyMatchFailure(op, "unsupported dialect");

    // For the ops that implement InferTypeOpInterface, we reinfer their return
    // types and see what happens.
    // Operands of these ops might have been refined elsewhere (e.g. someone
    // might have updated argument types of a function) or earlier during this
    // pass, and this might enable refinement opportunities downstream.
    SmallVector<Type> inferredReturnTypes;
    if (failed(op.inferReturnTypes(getContext(), /*location=*/{},
                                   op->getOperands(), op->getAttrDictionary(),
                                   op->getPropertiesStorage(), op->getRegions(),
                                   inferredReturnTypes)))
      return rewriter.notifyMatchFailure(op, "inferReturnTypes failed");
    return refineReturnTypes(rewriter, op, inferredReturnTypes);
  }
};

struct RefineRealDynamicSliceOpPattern
    : public OpRewritePattern<RealDynamicSliceOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(RealDynamicSliceOp op,
                                PatternRewriter& rewriter) const override {
    // Alternative #1: All attributes are fully static (SliceOp style).
    SmallVector<int64_t> startIndices, limitIndices, strides;
    if (succeeded(hlo::matchInts(op.getStartIndices(), startIndices)) &&
        succeeded(hlo::matchInts(op.getLimitIndices(), limitIndices)) &&
        succeeded(hlo::matchInts(op.getStrides(), strides))) {
      SmallVector<Type> inferredReturnTypes;
      if (failed(hlo::inferSliceOp(/*location=*/{}, op.getOperand().getType(),
                                   startIndices, limitIndices, strides,
                                   inferredReturnTypes)))
        return rewriter.notifyMatchFailure(op, "inferSliceOp failed");
      return refineReturnTypes(rewriter, op, inferredReturnTypes);
    }

    // Alternative #2: Slice sizes are fully static (DynamicSliceOp style).
    // To detect that, we check whether `limit_indices` is defined as
    // `start_indices + constant` or `constant + start_indices`.
    DenseIntElementsAttr sliceSizesAttr;
    auto m_startIndices = matchers::m_Val(op.getStartIndices());
    if (matchPattern(
            op.getLimitIndices(),
            m_Op<AddOp>(m_startIndices, m_Constant(&sliceSizesAttr))) ||
        matchPattern(
            op.getLimitIndices(),
            m_Op<AddOp>(m_Constant(&sliceSizesAttr), m_startIndices))) {
      SmallVector<int64_t> strides;
      if (!succeeded(hlo::matchInts(op.getStrides(), strides)) ||
          !llvm::all_of(strides, [&](int64_t stride) { return stride == 1; }))
        return rewriter.notifyMatchFailure(op, "expected unit strides");

      // RealDynamicSliceOp::start_indices is a 1-dimensional tensor.
      // DynamicSliceOp::start_indices is a vararg of 0-dimensional tensors.
      // Adapt accordingly in order to be compatible with inferDynamicSliceOp.
      auto startIndicesElementType =
          op.getStartIndices().getType().getElementType();
      SmallVector<Type> startIndicesTypes(
          sliceSizesAttr.size(),
          RankedTensorType::get({}, startIndicesElementType));

      // RealDynamicSliceOp can take tensors of integer or index element types.
      // DynamicSliceOp::slice_sizes only supports i64 element type.
      // Adapt accordingly in order to be compatible with inferDynamicSliceOp.
      SmallVector<int64_t> sliceSizes;
      for (auto element : sliceSizesAttr.getValues<APInt>()) {
        sliceSizes.push_back(element.getSExtValue());
      }

      SmallVector<ShapedTypeComponents> inferredReturnTypes;
      if (failed(hlo::inferDynamicSliceOp(
              op.getLoc(), op.getOperand().getType(), startIndicesTypes,
              rewriter.getDenseI64ArrayAttr(sliceSizes), inferredReturnTypes)))
        return rewriter.notifyMatchFailure(op, "inferDynamicSliceOp failed");
      return refineReturnTypes(rewriter, op, inferredReturnTypes);
    }

    return rewriter.notifyMatchFailure(
        op,
        "expected either fully static attributes (SliceOp style) "
        "or static sliceSizes (DynamicSliceOp style)");
  }
};

struct RefineReduceScatterOpPattern : public OpRewritePattern<ReduceScatterOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ReduceScatterOp op,
                                PatternRewriter& rewriter) const override {
    auto operandType = op.getOperand().getType();

    // This represents the cross_replica_and_partition process grouping strategy
    // that requires num_partitions to compute shardCount. Since we don't know
    // num_partitions at this point, we error out.
    if (op.getChannelHandle() && !op.getUseGlobalDeviceIds())
      return rewriter.notifyMatchFailure(op, "unsupported strategy");
    DenseIntElementsAttr replicaGroups = op.getReplicaGroups();
    auto shardCount = replicaGroups.getType().getDimSize(1);

    SmallVector<int64_t> refinement(operandType.getShape());
    if (!operandType.isDynamicDim(op.getScatterDimension()))
      refinement[op.getScatterDimension()] /= shardCount;
    return refineReturnShape(rewriter, op, refinement);
  }
};

struct RefineRngOpPattern : public OpRewritePattern<RngOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(RngOp op,
                                PatternRewriter& rewriter) const override {
    return refineReturnShape(rewriter, op, op.getShape());
  }
};

struct RefineUniformQuantizeOpPattern
    : public OpRewritePattern<UniformQuantizeOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(UniformQuantizeOp op,
                                PatternRewriter& rewriter) const override {
    SmallVector<ShapedTypeComponents> inferredReturnShapes;
    if (failed(hlo::inferUniformQuantizeOp(
            /*location=*/{}, op.getOperand(), inferredReturnShapes)))
      return rewriter.notifyMatchFailure(op, "inferConvertOp failed");
    return refineReturnTypes(rewriter, op, inferredReturnShapes);
  }
};

struct RefineWhileOpPattern : public OpRewritePattern<WhileOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(WhileOp op,
                                PatternRewriter& rewriter) const override {
    // Push the potentially refined operand types into the nested regions.
    // This can lead to refinements of the return types of the body (but not
    // of the cond since it always returns tensor<i1>), but the key insight here
    // is that the enclosing while op doesn't care about these refinements
    // (because its return types are equal to its operand types).
    // If we end up with incompatibilities between while's return types and
    // body's return types, the verifier will tell us about that. This means
    // that the original program wasn't well-formed. TODO(burmako): Implement
    // better error reporting for this case.
    // This serves the current use cases well, so the implementation of more
    // sophisticated refinement algorithm is left for future work.
    rewriter.startOpModification(op);
    auto condStatus = refineValues(rewriter, op, op.getCond().getArguments(),
                                   op.getOperandTypes());
    auto bodyStatus = refineValues(rewriter, op, op.getBody().getArguments(),
                                   op.getOperandTypes());
    if (succeeded(condStatus) || succeeded(bodyStatus)) {
      rewriter.finalizeOpModification(op);
      return success();
    } else {
      rewriter.cancelOpModification(op);
      return failure();
    }
  }
};

struct UpdateFunctionTypePattern : public OpRewritePattern<func::ReturnOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(func::ReturnOp op,
                                PatternRewriter& rewriter) const override {
    // Check whether any of the values returned by `func.return` are casts
    // which convert more specific type to less specific type.
    // Such ops are produced by the algorithm behind this pass to avoid
    // bringing the enclosing `func.func` op into an inconsistent state when
    // refining individual ops. This pattern cleans this up.
    bool needsUpdate = false;
    SmallVector<Type> updatedResultTypes(op.getOperandTypes());
    llvm::SmallSet<UnrealizedConversionCastOp, 4> castsToReplace;
    for (auto [i, operand] : llvm::enumerate(op.getOperands())) {
      auto cast =
          dyn_cast_or_null<UnrealizedConversionCastOp>(operand.getDefiningOp());
      if (!cast || cast.getInputs().size() != 1 ||
          cast.getOutputs().size() != 1)
        continue;

      // Only proceed if the type that we're casting from is more specific
      // than the type that we're casting to.
      auto sourceType = cast.getInputs()[0].getType();
      auto destType = cast.getOutputs()[0].getType();
      auto mostSpecificType = hlo::inferMostSpecificType(
          /*location=*/{}, {sourceType, destType});
      if (failed(mostSpecificType) || destType == *mostSpecificType) continue;

      // If the source type of the cast is more specific than the target type,
      // then we conclude that the cast is redundant (i.e. needs to be removed)
      // and that the return type of the function needs an update.
      needsUpdate = true;
      updatedResultTypes[i] = sourceType;

      // Insert into set and continue iterating.
      // ReturnOp may point to same value more than once.
      castsToReplace.insert(cast);
    }
    if (!needsUpdate)
      return rewriter.notifyMatchFailure(op, "doesn't need update");

    // Replace CastOps with more specific operands than results.
    for (auto cast : castsToReplace)
      rewriter.replaceOp(cast, cast->getOperands());

    // If the type of the enclosing `func.func` needs an update, we simply
    // call setType. We can afford this simplicity because our algorithm
    // currently supports only one function per module.
    auto func = cast<func::FuncOp>(op->getParentOp());
    func.setType(
        rewriter.getFunctionType(func.getArgumentTypes(), updatedResultTypes));
    return success();
  }
};

struct UpdateRegionTypePattern : public OpRewritePattern<ReturnOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ReturnOp op,
                                PatternRewriter& rewriter) const override {
    if (!isa<CaseOp, IfOp>(op->getParentOp()))
      return rewriter.notifyMatchFailure(op, "unsupported region");

    bool needsUpdate = false;
    SmallVector<Type> updatedResultTypes(op.getOperandTypes());
    for (auto [regionType, refinedType] : llvm::zip(
             op->getParentOp()->getResultTypes(), op->getOperandTypes())) {
      auto mostSpecificType = hlo::inferMostSpecificType(
          /*location=*/{}, {regionType, refinedType});
      if (failed(mostSpecificType) || regionType == *mostSpecificType) continue;
      needsUpdate = true;
    }
    if (!needsUpdate)
      return rewriter.notifyMatchFailure(op, "doesn't need update");

    rewriter.modifyOpInPlace(op->getParentOp(), [&]() { return; });
    return success();
  }
};

LogicalResult applyShapeRefinementPatterns(func::FuncOp func,
                                           RefineShapeState& state) {
  MLIRContext* context = func.getContext();
  RewritePatternSet patterns(func->getContext());
  GreedyRewriteConfig config;

  // The algorithm behind this pass consists of a single traversal of the
  // function. This is sufficient because we only support one function per
  // program at the moment.
  // TODO(#1048): Find out why .setMaxIterations(1) no longer works.
  // There have been recent refactors to applyPatternsGreedily
  // upstream, and that might be the reason.
  config.setUseTopDownTraversal(true)
      .setRegionSimplificationLevel(GreedySimplifyRegionLevel::Aggressive)
      .setMaxIterations(2)
      .setMaxNumRewrites(GreedyRewriteConfig::kNoLimit)
      .setStrictness(GreedyRewriteStrictness::AnyOp);

  populateStablehloRefineShapesPatterns(context, &patterns);
  patterns.add<RefineCallOpPattern>(context, state);

  // Populate additional patterns for StableHLO extensions.
  state.addAdditionalPatterns(patterns);

  StablehloAggressiveFolderPassOptions folderOptions;
  folderOptions.optimizeFloat = false;

  // The folding patterns implement partial evaluation of shape computations
  // which is a critical part of implementing type refinement for ops like
  // dynamic_broadcast_in_dim, dynamic_iota and dynamic_reshape whose shape
  // depends on the value of their shape operands.
  populateStablehloShapeFolderPatterns(context, &patterns, folderOptions);

  if (failed(applyPatternsGreedily(func, std::move(patterns), config)))
    func.emitError("Failed to converge StablehloRefineShapes in ")
        << config.getMaxIterations() << " iterations";

  return success();
}

LogicalResult refineFunction(MLIRContext& context, RefineShapeState& state,
                             RefinementKey& key) {
  LLVM_DEBUG(llvm::dbgs() << "Refining: " << key.toString() << "\n");
  auto refinementState = state.validateFunctionRefinement(key);
  if (failed(refinementState)) return failure();

  auto func = key.getFunc();
  if (*refinementState == RefineShapeState::RefinementState::ALREADY_REFINED) {
    LLVM_DEBUG(llvm::dbgs() << "Function " << func.getName()
                            << " already refined, skipping\n");
    return success();
  }

  auto scopedCleanup = state.createScopedFunctionRefinement(key);

  // StableHLO functions must have exactly one block.
  if (!func.getRegion().hasOneBlock())
    return func.emitOpError() << "must have exactly one block";

  // Replace the global-constant arguments with their values and drop args.
  // Wrap non-global-constant arguments with bitcast_convert.
  OpBuilder builder(func.getRegion());
  builder.setInsertionPointToStart(&func.getRegion().front());
  int64_t leadingTokenOperands = key.getLeadingTokenOperands();

  for (auto [i, dimValue] : llvm::enumerate(key.getGlobalConstants())) {
    int64_t operandIdx = leadingTokenOperands + i;
    BlockArgument arg = func.getArgument(operandIdx);
    Type argType = arg.getType();
    ShapedType argShapedType = dyn_cast<ShapedType>(argType);

    if (!argShapedType)
      return func.emitOpError()
             << "expected global constant argument to be shaped";

    auto replacement_op = builder.create<stablehlo::ConstantOp>(
        arg.getLoc(), argType, DenseElementsAttr::get(argShapedType, dimValue));
    arg.replaceAllUsesWith(replacement_op);
  }
  BitVector argIndices(func.getNumArguments());
  size_t firstFunctionalArgument =
      leadingTokenOperands + key.getGlobalConstants().size();
  argIndices.set(leadingTokenOperands, firstFunctionalArgument);
  if (failed(func.eraseArguments(argIndices))) return failure();

  // Refine the remaining argument types, wrap with shape buffer custom calls.
  SmallVector<Type> refinedTypes =
      key.getAllNonGlobalConstantArgumentTypes(context);
  if (failed(refineArguments(func, refinedTypes))) return failure();
  LLVM_DEBUG(llvm::dbgs() << "Refined function type for " << func.getName()
                          << ": " << func.getFunctionType() << "\n");

  // Now iterate into the function body and apply refinement patterns.
  if (failed(applyShapeRefinementPatterns(func, state))) return failure();

  LLVM_DEBUG(llvm::dbgs() << "refineFunction " << func.getName()
                          << ": end with type " << func.getFunctionType()
                          << "\n");
  return success();
}

struct StablehloRefineShapesPass
    : public impl::StablehloRefineShapesPassBase<StablehloRefineShapesPass> {
  using StablehloRefineShapesPassBase::StablehloRefineShapesPassBase;

  void runOnOperation() override {
    auto func = getStablehloRefineShapesTarget(getOperation());
    if (!func) return signalPassFailure();

    // Start with empty state, and no dim args / token args.
    MLIRContext* context = func.getContext();
    if (failed(refineEntryFunction(*context, func))) return signalPassFailure();
  }
};

}  // namespace

LogicalResult refineEntryFunction(
    MLIRContext& context, func::FuncOp func,
    std::optional<AdditionalShapeRefinementPatternsFn> additionalPatternsFn) {
  // Start with empty state, and no dim args / token args.
  RefineShapeState state(additionalPatternsFn);
  RefinementKey key(func, 0, {}, llvm::to_vector(func.getArgumentTypes()));
  if (failed(refineFunction(context, state, key)))
    return func.emitError("Failed to refine entry function");
  return success();
}

func::FuncOp getStablehloRefineShapesTarget(ModuleOp module) {
  // Only one function per module is supported at the moment to avoid the need
  // to think about iterative type inference algorithms.
  // Current use cases are served well by inlining multiple functions into
  // a single function, so we leave native support for multiple functions to
  // future work.
  // To enable modules that contain CustomCallOp::called_computations,
  // we allow multiple functions, in which case we only refine the main
  // function called "main", assuming that the called computations will have
  // static shapes. Lifting this assumption and expanding refinement to
  // multiple functions is left for future work.
  auto funcs = llvm::to_vector(module.getOps<func::FuncOp>());
  if (funcs.empty()) return nullptr;

  func::FuncOp result;
  if (funcs.size() == 1) {
    result = funcs[0];
  } else {
    result = module.lookupSymbol<func::FuncOp>("main");
  }
  if (!result) {
    module.emitOpError()
        << "must have no more than one function or a `main`"
        << " function to clearly identify which function will be refined";
    return nullptr;
  }

  // Similarly, only one block per function is supported at the moment.
  // At the StableHLO level, functions are expected to only have one block,
  // so supporting more is out of scope for this pass.
  if (!result.getRegion().hasOneBlock()) {
    result.emitOpError() << "must have exactly one block";
    return nullptr;
  }

  return result;
}

void populateStablehloRefineShapesPatterns(MLIRContext* context,
                                           RewritePatternSet* patterns) {
  patterns->add<RefineAllGatherOpPattern>(context);
  patterns->add<RefineBitcastConvertOpPattern>(context);
  // patterns->add<RefineCallOpPattern>(context); // Populate requires inline
  patterns->add<RefineConvertOpPattern>(context);
  patterns->add<RefineConvolutionOpPattern>(context);
  patterns->add<RefineCustomCallOpPattern>(context);
  patterns->add<RefineDotGeneralOpPattern>(context);
  patterns->add<RefineDotOpPattern>(context);
  patterns->add<RefineDynamicBroadcastInDimOpPattern>(context);
  patterns->add<RefineDynamicConvOpPattern>(context);
  patterns->add<RefineDynamicIotaOpPattern>(context);
  patterns->add<RefineDynamicPadOpPattern>(context);
  patterns->add<RefineDynamicReshapeOpPattern>(context);
  patterns->add<RefineInferTypeOpInterfacePattern>(context);
  patterns->add<RefineRealDynamicSliceOpPattern>(context);
  patterns->add<RefineReduceScatterOpPattern>(context);
  patterns->add<RefineRngOpPattern>(context);
  patterns->add<RefineUniformQuantizeOpPattern>(context);
  patterns->add<RefineWhileOpPattern>(context);
  patterns->add<UpdateFunctionTypePattern>(context);
  patterns->add<UpdateRegionTypePattern>(context);
}

}  // namespace stablehlo
}  // namespace mlir
