/* Copyright 2022 OpenXLA Authors. All Rights Reserved.

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

include "mlir/Pass/PassBase.td"

def StablehloLegalizeToTosaPass : Pass<"stablehlo-legalize-to-tosa", "mlir::func::FuncOp"> {
  let summary = "Legalize StableHLO to TOSA";
  let dependentDialects = [
    "::mlir::pdl::PDLDialect",
    "::mlir::pdl_interp::PDLInterpDialect",
    "::mlir::tosa::TosaDialect",
  ];
}

def StablehloPrepareForTosaPass : Pass<"stablehlo-prepare-for-tosa", "mlir::func::FuncOp"> {
  let summary = "Prepare StableHLO for legalization to TOSA";
  let description = [{
    This pass adds rewriters to make StableHLO ops more compatible with TOSA ops.
    Currently simplifies stablehlo.dot_general into stablehlo.dot for easier lowering.
  }];
  let dependentDialects = ["::mlir::tosa::TosaDialect"];
}

def StablehloQuantLegalizeToTosaRescalePass : Pass<"stablehlo-quant-legalize-to-tosa-rescale", "mlir::func::FuncOp"> {
  let summary = "Legalize StableHLO Quantized operations to TOSA rescale operations";
  let description = [{
    This pass rewrites StableHLO quantized operations to integer operations
    by inserting TOSA rescale operations at the inputs and outputs of the
    integer operations.
  }];
  let dependentDialects = [
    "::mlir::tosa::TosaDialect",
  ];
}

def TosaRescaleLegalizeToStablehloPass : Pass<"tosa-rescale-legalize-to-stablehlo", "mlir::func::FuncOp"> {
  let summary = "Legalize TOSA rescales to StableHlo primitive math operations";
  let description = [{
    This pass rewrites TOSA rescale operations to StableHLO primitive math operations.
  }];
  let dependentDialects = [
    "::mlir::stablehlo::StablehloDialect"
  ];
}
