/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.
   Copyright 2023 The StableHLO Authors.

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

#ifndef STABLEHLO_DIALECT_STABLEHLO_TYPES
#define STABLEHLO_DIALECT_STABLEHLO_TYPES

include "mlir/IR/AttrTypeBase.td"

class StableHLO_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<StableHLO_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

// Corresponds to TokenType from the StableHLO spec.
def StableHLO_Token : StableHLO_Type<"Token", "token">;

#endif  // STABLEHLO_DIALECT_STABLEHLO_TYPES
