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
