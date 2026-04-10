/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#ifndef STABLEHLO_TRANSFORMS_REPLICA_GROUP_UTILS_H
#define STABLEHLO_TRANSFORMS_REPLICA_GROUP_UTILS_H

#include <cstdint>

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {
namespace stablehlo {

// Flattens replica groups represented by a mesh and a list of communication
// axes into a flat list of replica groups.
// Returns a 2D array of device IDs representing the flattened replica
// groups, or failure if the conversion fails.
FailureOr<SmallVector<SmallVector<int64_t>>> flattenReplicaGroupMeshAxes(
    Attribute meshAttr, ArrayAttr commAxes, Location loc);

}  // namespace stablehlo
}  // namespace mlir

#endif  // STABLEHLO_TRANSFORMS_REPLICA_GROUP_UTILS_H
