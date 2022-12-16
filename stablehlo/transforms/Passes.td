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

include "mlir/Pass/PassBase.td"

def StablehloLegalizeToVhloPass : Pass<"stablehlo-legalize-to-vhlo", "ModuleOp"> {
  let summary = "Legalize StableHLO to VHLO.";
  let dependentDialects = ["mlir::vhlo::VhloDialect"];
}

def VhloLegalizeToStablehloPass : Pass<"vhlo-legalize-to-stablehlo", "ModuleOp"> {
  let summary = "Legalize VHLO to StableHLO.";
  let dependentDialects = ["mlir::stablehlo::StablehloDialect"];
}

def VhloToVersionPass : Pass<"vhlo-to-version"> {
  let summary = "Convert between versions of VHLO.";
  let options = [
    Option<"targetVersionOption", "target", "std::string", "",
           "The target version. Must be a version of the form #.#.# or 'current'.">,
  ];
}
