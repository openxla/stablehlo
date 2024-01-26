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

// This is the utils file for the StableHlo dialect.

#ifndef STABLEHLO_DIALECT_STABLEHLO_UTILS
#define STABLEHLO_DIALECT_STABLEHLO_UTILS

include "mlir/IR/OpBase.td"
// Constraint that Attr has values [0, 1, ...].
def IdentityBroadcastDims : AttrConstraint<
    CPred<"hlo::isSequenceStartingWith0($_self)">>;

#endif  // STABLEHLO_DIALECT_STABLEHLO_UTILS 
