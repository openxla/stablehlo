/* Copyright 2020 OpenXLA Authors. All Rights Reserved.

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
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "stablehlo/dialect/Register.h"
#include "transforms/passes.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  mlir::tosa::registerTosaLegalizeStablehloPassPass();
  mlir::tosa::registerTosaPrepareStablehloPassPass();

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::stablehlo::registerAllDialects(registry);

  return failed(
      mlir::MlirOptMain(argc, argv, "Stablehlo TOSA pass driver\n", registry));
}
