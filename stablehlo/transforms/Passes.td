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

def StablehloCanonicalizeDynamismPass : Pass<"stablehlo-canonicalize-dynamism", "func::FuncOp"> {
  let summary = "Canonicalizes dynamic StableHLO ops into static ops.";
  let description = [{
    Replaces dynamic StableHLO ops like DynamicReshapeOp with the corresponding
    static counterparts like ReshapeOp if all the dynamic elements of these ops
    are actually constant.

    For example, if the output_shape operand of DynamicReshapeOp is a constant
    value, then the operation can be transformed to ReshapeOp.
  }];
}

def StablehloInstrumentWithProbePass : Pass<"stablehlo-instrument-with-probe", "ModuleOp"> {
  let summary = "Inserts probe instrumentation instructions in a StableHLO "
    "program.";
  let options = [
    Option<"useDebugInfoOption", "useDebugInfo", "bool", /*default=*/"false",
           "Whether or not to use location debug data as `probe_id` values.">,
  ];

  let dependentDialects = ["mlir::stablehlo::interpreter::InterpreterDialect"];
  let description = [{
    Walks through a StableHLO program and inserts a probe instrumentation
    operation after each suitable operation (see below for how a suitable
    operation is defined). Instrumentation is used to extract intermediate
    tensor values from the StableHLO reference interpreter for later comparison
    with other runtimes.

    All operations are considered suitable for instrumentation, except constant
    ops, ops which do not have any tensor return values (i.e. an op that
    produces a tuple or a token or no return values will not be instrumented).
    Suitable operations will be instrumented regardless of their level of
    nesting. That is, operations inside loop/branch regions will also be
    instrumented.
    
    Instrumented operations will have their return values written to disk using
    the NumPy data format as they are executed. If the `useDebugInfo` pass
    option is enabled, location debug information will be used when available to
    uniquely identify instrumented tensor values (i.e. the pass will extract
    `probe_id` from `NamedLoc(probe_id@<...>)` and use the format `probe_id`.#).
    Otherwise, instrumented values will be referred to in the increasing
    sequence: `probe1`, `probe2`, ... See `interpreter.probe` for additional
    information on how data is serialized.
  }];
}

def StablehloLegalizeToVhloPass : Pass<"stablehlo-legalize-to-vhlo", "ModuleOp"> {
  let summary = "Legalize StableHLO to VHLO.";
  let dependentDialects = ["mlir::vhlo::VhloDialect"];
}

def StablehloRefineShapesPass : Pass<"stablehlo-refine-shapes", "ModuleOp"> {
  let summary = "Refines shapes across a StableHLO program.";
  let description = [{
    Walks through a StableHLO program refining shapes within ops.

    The flagship use case for this pass is specializing dynamically-shaped
    programs to static shapes. If a dynamically-shaped StableHLO program has the
    right structure, then updating its argument types from dynamic shapes to
    static shapes and running this pass will propagate static shapes across
    the program.
  }];
}

def VhloLegalizeToStablehloPass : Pass<"vhlo-legalize-to-stablehlo", "ModuleOp"> {
  let summary = "Legalize VHLO to StableHLO.";
  let dependentDialects = ["mlir::func::FuncDialect", "mlir::stablehlo::StablehloDialect",
                           "mlir::shape::ShapeDialect", "mlir::quant::QuantizationDialect"];
}

def VhloToVersionPass : Pass<"vhlo-to-version"> {
  let summary = "Convert between versions of VHLO.";
  let options = [
    Option<"targetVersionOption", "target", "std::string", "",
           "The target version. Must be a version of the form #.#.# or 'current'.">,
  ];
}
