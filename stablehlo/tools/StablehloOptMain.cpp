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

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "stablehlo/conversions/tosa/transforms/Passes.h"
#include "stablehlo/dialect/Register.h"
#include "stablehlo/tests/TestUtils.h"
#include "stablehlo/transforms/Passes.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  mlir::hlo::registerAllTestPasses();
  mlir::stablehlo::registerPasses();
  mlir::tosa::registerStablehloLegalizeToTosaPassPass();
  mlir::tosa::registerStablehloPrepareForTosaPassPass();

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);
  mlir::stablehlo::registerAllDialects(registry);

  return failed(
      mlir::MlirOptMain(argc, argv, "StableHLO optimizer driver\n", registry));
}
