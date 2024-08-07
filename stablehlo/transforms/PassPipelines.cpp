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

#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"
#include "stablehlo/dialect/Version.h"
#include "stablehlo/transforms/Passes.h"

namespace mlir {
namespace stablehlo {

void createStablehloDeserializePipeline(OpPassManager &pm) {
  // Convert VHLO(version x.y.z) --> VHLO(current).
  pm.addPass(stablehlo::createVhloToVersionPass(
      {vhlo::Version::getCurrentVersion().toString()}));

  // Convert VHLO --> StableHLO. Will not fail within compatibility window.
  pm.addPass(stablehlo::createVhloLegalizeToStablehloPass());
}

void createChloPreSerializationPipeline(OpPassManager &pm) {
  // CHLO --> StableHLO+Shape --> StableHLO
  pm.addPass(stablehlo::createChloLegalizeToStablehloPass());
  pm.addPass(stablehlo::createShapeLegalizeToStablehloPass());
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());
}

void createStablehloRemoveDynamismPipeline(OpPassManager &pm,
                                           TypeRange refinedTypes) {
  pm.addPass(stablehlo::createStablehloRefineArgumentsPass(refinedTypes));
  pm.addPass(stablehlo::createStablehloRefineShapesPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      stablehlo::createStablehloCanonicalizeDynamismPass());
}

void createStablehloLowerQuantPipeline(OpPassManager &pm) {
  pm.addNestedPass<mlir::func::FuncOp>(
      stablehlo::createStablehloLegalizeQuantToMathPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      stablehlo::createChloLegalizeToStablehloPass());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      stablehlo::createShapeLegalizeToStablehloPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      stablehlo::createStablehloCanonicalizeDynamismPass());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
}

void registerPassPipelines() {
  PassPipelineRegistration<>("stablehlo-deserialize",
                             "Run an example pipeline.",
                             createStablehloDeserializePipeline);
  PassPipelineRegistration<>("chlo-pre-serialization-pipeline",
                             "Lower as much as possible to StableHLO, "
                             "including CHLO, Shape dialect, and index types.",
                             createChloPreSerializationPipeline);
}

}  // namespace stablehlo
}  // namespace mlir
