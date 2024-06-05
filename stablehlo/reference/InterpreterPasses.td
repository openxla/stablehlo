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

include "mlir/Pass/PassBase.td"

def InterpreterInstrumentWithProbePass : Pass<"interpreter-instrument-with-probe", "ModuleOp"> {
  let summary = "Inserts interpreter.probe ops in a StableHLO program.";
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
