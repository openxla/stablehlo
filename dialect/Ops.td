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

#ifndef STABLEHLO_OPS
#define STABLEHLO_OPS

include "mlir/IR/OpBase.td"

def StableHLO_Dialect : Dialect {
  let name = "stablehlo";
  let description = [{
    The StableHLO dialect.

    This dialect implements the StableHLO specification defined at
    https://github.com/openxla/stablehlo.
  }];
  let cppNamespace = "::mlir::stablehlo";
}

class StableHLO_Op<string mnemonic, list<Trait> traits> :
    Op<StableHLO_Dialect, mnemonic, traits> {
}

// TODO: Bootstrap the StableHLO dialect from the MHLO dialect,
// and start working on a plan for providing compatibility guarantees.
// See GitHub Discussions for more details.
def StableHLO_DummyOp : StableHLO_Op<"dummy", []> {
  let summary = "Dummy op, to be replaced once the dialect is bootstrapped";
  let results = (outs AnyTensor);
}

#endif // STABLEHLO_OPS
