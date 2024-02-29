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

#include <memory>
#include <string>
#include <vector>

#include "llvm/ADT/STLExtras.h"
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
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "stablehlo/dialect/Base.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/dialect/TypeInference.h"
#include "stablehlo/transforms/Passes.h"
#include "stablehlo/transforms/StablehloRefineShapes.h"

namespace mlir {
namespace stablehlo {

#define GEN_PASS_DEF_STABLEHLOREFINEARGUMENTSPASS
#include "stablehlo/transforms/Passes.h.inc"

namespace {

CustomCallOp makeShapeRefinementOperandWrapper(OpBuilder& builder,
                                               Value operand,
                                               Type refinedType) {
  auto rankedTensor = refinedType.cast<RankedTensorType>();
  auto constant = builder.create<stablehlo::ConstantOp>(
      operand.getLoc(), builder.getI64TensorAttr(rankedTensor.getShape()));
  return builder.create<stablehlo::CustomCallOp>(
      operand.getLoc(), operand.getType(), ValueRange{operand, constant},
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

// Validates refinement types:
//   - A type refinement must be specified for each operand
//   - Refinement types that match operand types are skipped
//   - Refinement types that do not match operands must be refining tensors
//   - Refined tensor types must be ranked, operand type can be unranked
//   - Refined tensor types must match operand type for all static dimensions
//
LogicalResult validateRefinedTypes(func::FuncOp func, TypeRange refinedTypes) {
  // Validate refined shapes
  if (func.getNumArguments() != refinedTypes.size()) {
    return func.emitOpError(
               "number of refinements must match number of function operands ")
           << refinedTypes.size() << " vs " << func.getNumArguments();
  }

  // Validate that refinements are valid
  for (auto [type, refinedType] :
       llvm::zip(func.getArgumentTypes(), refinedTypes)) {
    // Always allow skipping refinement
    if (type == refinedType) continue;

    // If mismatched, must be tensor types
    auto tensorType = type.dyn_cast<TensorType>();
    auto refinedTensorType = refinedType.dyn_cast<TensorType>();
    if (!tensorType || !refinedTensorType) {
      return func.emitOpError("refinement type must be tensor ")
             << type << " -> " << refinedType;
    }

    // Refined rank cannot be unranked if mismatch
    if (refinedType.isa<UnrankedTensorType>()) {
      return func.emitOpError("refinement must be ranked ")
             << type << " -> " << refinedType;
    }

    if (tensorType.hasRank()) {
      // Validate ranks match if ranked (must allow unranked tensorType)
      if (tensorType.getRank() != refinedTensorType.getRank()) {
        return func.emitOpError("refinement rank must match operand rank ")
               << type << " -> " << refinedType;
      }

      // Validate static dimension sizes match
      for (auto [dimSize, refinedDimSize] :
           llvm::zip(tensorType.getShape(), refinedTensorType.getShape())) {
        if (!ShapedType::isDynamic(dimSize) && dimSize != refinedDimSize) {
          return func.emitOpError(
                     "refinement dimension sizes must match for static "
                     "dimensions ")
                 << type << " -> " << refinedType;
        }
      }
    }
  }
  return success();
}

// Wrap operands in "type barriers" so the rest of the program remains valid
// after the signature update and before shape refinement
//
// %1 = stablehlo.custom_call
//   @stablehlo.shape_refinement_operand_wrapper(%arg1, %0)
//     {indices_of_shape_operands = dense<1> : tensor<1xi64>}
//     : (tensor<5x10xf32>, tensor<2xi64>) -> tensor<?x10xf32>
//
// When introduced this is an identity operation from:
//   (tensor<?x10xf32>) -> tensor<?x10xf32>
// But after the operand is changed, this op has type signature
//  (tensor<5x10xf32>) -> tensor<?x10xf32>
//
// Before shape refinement, all future uses of this argument expect type
// tensor<?x10xf32>. By updating these uses to instead use the wrapper, the IR
// remains valid in the intermediate state.
void wrapRefinedOperands(func::FuncOp func, TypeRange refinedTypes) {
  Region& body = func.getBody();
  OpBuilder builder(body);
  builder.setInsertionPointToStart(&body.front());
  for (int i = 0; i < static_cast<int>(body.getNumArguments()); ++i) {
    BlockArgument arg = body.getArgument(i);
    Type argType = arg.getType();
    Type refinedType = refinedTypes[i];
    if (argType != refinedType) {
      auto customCall =
          makeShapeRefinementOperandWrapper(builder, arg, refinedType);
      auto callResult = customCall.getResult(0);
      arg.replaceAllUsesExcept(callResult, customCall);
    }
  }
}

void refineOperandsAndUpdateFunctionSignature(func::FuncOp func,
                                              TypeRange refinedTypes) {
  Region& body = func.getBody();
  OpBuilder builder(body);
  for (int i = 0; i < static_cast<int>(body.getNumArguments()); ++i) {
    auto arg = body.getArgument(i);
    arg.setType(refinedTypes[i]);
  }
  func.setType(builder.getFunctionType(refinedTypes, func.getResultTypes()));
}

struct StablehloRefineArgumentsPass
    : public impl::StablehloRefineArgumentsPassBase<
          StablehloRefineArgumentsPass> {
  using Super =
      impl::StablehloRefineArgumentsPassBase<StablehloRefineArgumentsPass>;
  StablehloRefineArgumentsPass() : Super() {}
  StablehloRefineArgumentsPass(
      const StablehloRefineArgumentsPassOptions& opts)
      : Super(opts) {}
  StablehloRefineArgumentsPass(TypeRange refinedTypes_) : Super() {
    refinedTypes = llvm::to_vector(refinedTypes_);
  }

  void runOnOperation() override {
    auto func = getStablehloRefineShapesTarget(getOperation());
    if (!func) return signalPassFailure();

    // Parse if string specified as option
    if (!refinedShapesOption.empty() &&
        failed(parseRefinedTypes(getOperation(), refinedShapesOption,
                                 refinedTypes))) {
      return signalPassFailure();
    }

    // Verify that refinements are valid
    if (failed(validateRefinedTypes(func, refinedTypes)))
      return signalPassFailure();

    // Wrap refined operands in operand wrapper to keep IR valid for refinement
    wrapRefinedOperands(func, refinedTypes);

    // Actually update main's input types.
    refineOperandsAndUpdateFunctionSignature(func, refinedTypes);
  }

 private:
  SmallVector<Type> refinedTypes;
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> createStablehloRefineArgumentsPass(
    TypeRange refinedTypes) {
  return std::make_unique<StablehloRefineArgumentsPass>(refinedTypes);
}
}  // namespace stablehlo
}  // namespace mlir
