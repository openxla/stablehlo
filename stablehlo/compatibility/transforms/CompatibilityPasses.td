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

def StablehloLegalizeToVersionedhloPass : Pass<"stablehlo-legalize-to-versionedhlo", "ModuleOp"> {
  let summary = "Legalize StableHLO to VersionedStableHLO.";
  let dependentDialects = ["mlir::versionedhlo::VersionedhloDialect"];
}

def VersionedhloLegalizeToStablehloPass : Pass<"versionedhlo-legalize-to-stablehlo", "ModuleOp"> {
  let summary = "Legalize VersionedStableHLO to StableHLO.";
  let dependentDialects = ["mlir::stablehlo::StablehloDialect"];
}

def VersionedhloUpgradePass : Pass<"versionedhlo-upgrade", "ModuleOp"> {
  let summary = "Upgrade Versioned StableHLO.";
}

def VersionedhloDowngradePass : Pass<"versionedhlo-downgrade", "ModuleOp"> {
  let summary = "Downgrade Versioned StableHLO to target version.";
  let options = [
    Option<"targetVersion", "target-version", "int64_t", /*default=*/"0",
           "The target version for downgrade. (Defaults to minimum supported)">,
  ];
}