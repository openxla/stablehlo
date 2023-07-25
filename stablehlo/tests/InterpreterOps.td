/* Copyright 2023 The StableHLO Authors.

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

#ifndef STABLEHLO_TESTS_INTERPRETER_OPS
#define STABLEHLO_TESTS_INTERPRETER_OPS

include "stablehlo/dialect/Base.td"
include "stablehlo/tests/InterpreterDialect.td"

//===----------------------------------------------------------------------===//
// Interpreter op definitions.
//===----------------------------------------------------------------------===//

def Interpreter_RunParallelOp : Op<Interpreter_Dialect, "run_parallel", []> {
  let arguments = (ins
    Variadic<HLO_TensorOrToken>:$inputs,
    StrArrayAttr:$programs,
    I64Attr:$num_replicas,
    I64Attr:$num_partitions
  );
  let results = (outs Variadic<HLO_TensorOrToken>:$results);
}

#endif  // STABLEHLO_TESTS_INTERPRETER_OPS
