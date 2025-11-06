/* Copyright 2024 The StableHLO Authors.
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

#include <cstdint>
#include <memory>
#include <string>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/transforms/Passes.h"
#include "stablehlo/transforms/StablehloRefineShapes.h"

namespace mlir {
namespace stablehlo {

#define GEN_PASS_DEF_STABLEHLOREFINEARGUMENTSPASS
#include "stablehlo/transforms/Passes.h.inc"

namespace {

CustomCallOp makeShapeRefinementOperandWrapper(OpBuilder& builder,
                                               Value operand,
                                               RankedTensorType refinedType) {
  auto constant = stablehlo::ConstantOp::create(
      builder, operand.getLoc(),
      builder.getI64TensorAttr(refinedType.getShape()));
  return stablehlo::CustomCallOp::create(
      builder, operand.getLoc(), operand.getType(),
      ValueRange{operand, constant},
      llvm::SmallVector<NamedAttribute>{
          builder.getNamedAttr(
              "call_target_name",
              builder.getStringAttr(kCustomCallOperandBarrierTarget)),
          builder.getNamedAttr("indices_of_shape_operands",
                               builder.getI64TensorAttr({1}))});
}

ParseResult parseRefinedTypes(ModuleOp module,
                              ArrayRef<std::string> shapeString,
                              SmallVector<Type>& refinedTypes) {
  MLIRContext* context = module.getContext();
  for (const auto& shape : shapeString) {
    Type type = mlir::parseType(shape, context);
    if (!type) return module->emitOpError("Invalid type string: ") << shape;
    refinedTypes.push_back(type);
  }
  return success();
}

// Wrap operands in "type barriers" so the rest of the program remains valid
// after the signature update and before shape refinement.
//
// %0 = stablehlo.constant dense<[10, 5]> : tensor<2xi64>
// %1 = stablehlo.custom_call
//   @stablehlo.shape_refinement_operand_wrapper(%arg1, %0)
//     {indices_of_shape_operands = dense<1> : tensor<1xi64>}
//     : (tensor<5x10xf32>, tensor<2xi64>) -> tensor<?x10xf32>
//
// Before shape refinement, all future uses of this argument expect type
// tensor<?x10xf32>. By updating these uses to instead use the wrapper, the IR
// remains valid in the intermediate state.
void wrapRefinedOperands(func::FuncOp func, TypeRange refinedTypes) {
  Region& body = func.getBody();
  OpBuilder builder(body);
  builder.setInsertionPointToStart(&body.front());
  for (int64_t i = 0; i < body.getNumArguments(); ++i) {
    BlockArgument arg = body.getArgument(i);
    Type argType = arg.getType();
    Type refinedType = refinedTypes[i];
    if (argType != refinedType) {
      auto rankedRefinedType = cast<RankedTensorType>(refinedType);
      auto customCall =
          makeShapeRefinementOperandWrapper(builder, arg, rankedRefinedType);
      auto callResult = customCall.getResult(0);
      arg.replaceAllUsesExcept(callResult, customCall);
    }
  }
}

void refineOperandsAndUpdateFunctionSignature(func::FuncOp func,
                                              TypeRange refinedTypes) {
  Region& body = func.getBody();
  OpBuilder builder(body);
  for (int64_t i = 0; i < body.getNumArguments(); ++i) {
    auto arg = body.getArgument(i);
    arg.setType(refinedTypes[i]);
  }
  func.setType(builder.getFunctionType(refinedTypes, func.getResultTypes()));
}

struct StablehloRefineArgumentsPass
    : public impl::StablehloRefineArgumentsPassBase<
          StablehloRefineArgumentsPass> {
  using StablehloRefineArgumentsPassBase::StablehloRefineArgumentsPassBase;

  StablehloRefineArgumentsPass(TypeRange refinedTypes_)
      : StablehloRefineArgumentsPassBase() {
    refinedTypes = llvm::to_vector(refinedTypes_);
  }

  void runOnOperation() override {
    auto func = getStablehloRefineShapesTarget(getOperation());
    if (!func) return signalPassFailure();

    // Parse if string specified as option
    if (!refinedTypesOption.empty() &&
        failed(parseRefinedTypes(getOperation(), refinedTypesOption,
                                 refinedTypes))) {
      return signalPassFailure();
    }

    if (failed(refineArguments(func, refinedTypes))) return signalPassFailure();
  }

 private:
  SmallVector<Type> refinedTypes;
};

}  // namespace

LogicalResult refinementError(Operation* op, int64_t idx, Type argType,
                              Type refinedType, StringRef msg) {
  return op->emitOpError() << "invalid refinement for argument " << idx
                           << ", refinement " << msg << " in "
                           << mlir::debugString(argType) << " -> "
                           << mlir::debugString(refinedType);
}

LogicalResult validateRefinedTypes(Operation* op, TypeRange argTypes,
                                   TypeRange refinedTypes) {
  // Validate refined shapes
  if (argTypes.size() != refinedTypes.size()) {
    return op->emitOpError(
               "number of refinements must match number of op operands ")
           << refinedTypes.size() << " vs " << argTypes.size();
  }

  // Validate that refinements are valid
  for (size_t i = 0; i < argTypes.size(); ++i) {
    Type type = argTypes[i];
    Type refinedType = refinedTypes[i];

    // Always allow skipping refinement
    if (type == refinedType) continue;

    // If mismatched, must be tensor types
    auto tensorType = dyn_cast<TensorType>(type);
    auto refinedTensorType = dyn_cast<TensorType>(refinedType);
    if (!tensorType || !refinedTensorType) {
      return refinementError(op, i, type, refinedType, "must be a tensor");
    }

    // Check that element types match
    if (tensorType.getElementType() != refinedTensorType.getElementType()) {
      return refinementError(op, i, type, refinedType,
                             "element types must match");
    }

    // Refined rank cannot be unranked if mismatch
    if (isa<UnrankedTensorType>(refinedType)) {
      return refinementError(op, i, type, refinedType, "must be ranked");
    }

    // Unranked operands can be refined to anything
    if (!tensorType.hasRank()) continue;

    // Validate ranks match if ranked (must allow unranked tensorType)
    if (tensorType.getRank() != refinedTensorType.getRank()) {
      return refinementError(op, i, type, refinedType,
                             "rank must match operand rank");
    }

    // Validate static dimension sizes match
    for (auto [dimSize, refinedDimSize] :
         llvm::zip(tensorType.getShape(), refinedTensorType.getShape())) {
      if (!ShapedType::isDynamic(dimSize) && dimSize != refinedDimSize) {
        return refinementError(
            op, i, type, refinedType,
            "dimension sizes must match for static dimensions");
      }
    }
  }
  return success();
}

LogicalResult refineArguments(func::FuncOp func, TypeRange refinedTypes) {
  // Verify that refinements are valid
  if (failed(validateRefinedTypes(func, func.getArgumentTypes(), refinedTypes)))
    return failure();

  // Wrap refined operands in operand wrapper to keep IR valid for refinement
  wrapRefinedOperands(func, refinedTypes);

  // Actually update main's input types.
  refineOperandsAndUpdateFunctionSignature(func, refinedTypes);

  return success();
}

std::unique_ptr<OperationPass<ModuleOp>> createStablehloRefineArgumentsPass(
    TypeRange refinedTypes) {
  return std::make_unique<StablehloRefineArgumentsPass>(refinedTypes);
}
}  // namespace stablehlo
}  // namespace mlir
