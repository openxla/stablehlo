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

#include "stablehlo/dialect/ReplicaGroupUtils.h"

#include <cstddef>
#include <cstdint>
#include <utility>

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {
namespace stablehlo {

static SmallVector<SmallVector<int64_t>>
flattenedReplicaGroupsFromTransposePermutation(
    const SmallVector<StringRef>& meshAxisNames,
    const SmallVector<StringRef>& commAxisNames,
    const llvm::DenseSet<StringRef>& commAxisSet,
    const SmallVector<int64_t>& axisSizes,
    const SmallVector<int64_t>& deviceIds, int64_t totalDevices) {
  // Reshape and Transpose equivalence bridging XLA TileAssignment behavior.
  SmallVector<int64_t> transposeAxes;
  // Non-grouped axes first
  for (size_t i = 0; i < meshAxisNames.size(); ++i) {
    if (!commAxisSet.count(meshAxisNames[i])) {
      transposeAxes.push_back(i);
    }
  }
  // Grouped axes
  for (const auto& name : commAxisNames) {
    for (size_t i = 0; i < meshAxisNames.size(); ++i) {
      if (meshAxisNames[i] == name) {
        transposeAxes.push_back(i);
        break;
      }
    }
  }

  SmallVector<int64_t> transposedSizes(meshAxisNames.size());
  for (size_t i = 0; i < meshAxisNames.size(); ++i) {
    transposedSizes[i] = axisSizes[transposeAxes[i]];
  }

  // Compute strides for original shape
  SmallVector<int64_t> originalStrides(meshAxisNames.size(), 1);
  for (int i = static_cast<int>(meshAxisNames.size()) - 2; i >= 0; --i) {
    originalStrides[i] = originalStrides[i + 1] * axisSizes[i + 1];
  }

  // Compute strides for transposed shape
  SmallVector<int64_t> transposedStrides(meshAxisNames.size(), 1);
  for (int i = static_cast<int>(meshAxisNames.size()) - 2; i >= 0; --i) {
    transposedStrides[i] = transposedStrides[i + 1] * transposedSizes[i + 1];
  }

  // Generate chunks
  int64_t numDevicesPerGroup = 1;
  for (auto name : commAxisNames) {
    for (size_t i = 0; i < meshAxisNames.size(); ++i) {
      if (meshAxisNames[i] == name) {
        numDevicesPerGroup *= axisSizes[i];
        break;
      }
    }
  }
  int64_t numGroups = totalDevices / numDevicesPerGroup;

  SmallVector<SmallVector<int64_t>> groups;
  groups.reserve(numGroups);
  for (int64_t i = 0; i < numGroups; ++i) {
    SmallVector<int64_t> group;
    group.reserve(numDevicesPerGroup);
    for (int64_t j = 0; j < numDevicesPerGroup; ++j) {
      int64_t linearTransposeIdx = i * numDevicesPerGroup + j;
      int64_t originalIndex = 0;
      for (size_t k = 0; k < meshAxisNames.size(); ++k) {
        int64_t coord =
            (linearTransposeIdx / transposedStrides[k]) % transposedSizes[k];
        originalIndex += coord * originalStrides[transposeAxes[k]];
      }
      group.push_back(deviceIds[originalIndex]);
    }
    groups.push_back(std::move(group));
  }

  return groups;
}

FailureOr<SmallVector<SmallVector<int64_t>>> flattenReplicaGroupMeshAxes(
    Attribute meshAttr, ArrayAttr commAxes, Location loc) {
  // Reference implementation for flattening a mesh-axes based replica group
  // to the list of lists implementation.
  //
  // Note, that this function is only intended for reference implementation
  // purposes or when a fallback is required in VHLO legalization if the current
  // version doesn't support mesh-axes based replica groups. In all other cases
  // we don't flatten mesh-axes based replica groups, since we intend to persist
  // them throughout the pipeline.
  auto mesh = dyn_cast<stablehlo::MeshAttr>(meshAttr);
  if (!mesh)
    return emitOptionalError(loc, "expected stablehlo.mesh for mesh attribute");

  auto axesInMesh = mesh.getAxes();

  // Identify which axes are communication axes.
  llvm::SmallVector<StringRef> commAxisNames;
  llvm::DenseSet<StringRef> commAxisSet;
  for (auto attr : commAxes) {
    auto shloAxisRef = llvm::dyn_cast<AxisRefAttr>(attr);
    if (!shloAxisRef) {
      return emitError(loc) << "expected AxisRefAttr in comm_axes";
    }
    if (shloAxisRef.getSubAxisInfo()) {
      return emitError(loc) << "Subaxes are not supported in "
                               "flattenReplicaGroupMeshAxes";
    }
    commAxisNames.push_back(shloAxisRef.getName());
    commAxisSet.insert(shloAxisRef.getName());
  }

  // Calculate total devices and axis sizes

  int64_t totalDevices = 1;
  SmallVector<int64_t> axisSizes;
  SmallVector<StringRef> meshAxisNames;
  for (auto meshAxis : axesInMesh) {
    auto typedMeshAxis = llvm::cast<stablehlo::MeshAxisAttr>(meshAxis);
    axisSizes.push_back(typedMeshAxis.getSize());
    meshAxisNames.push_back(typedMeshAxis.getName());
    totalDevices *= typedMeshAxis.getSize();
  }

  SmallVector<int64_t> deviceIds;
  if (!mesh.getDeviceIds() || mesh.getDeviceIds().empty()) {
    for (int64_t i = 0; i < totalDevices; ++i) deviceIds.push_back(i);
  } else {
    for (auto id : mesh.getDeviceIds().getValues<int64_t>()) {
      deviceIds.push_back(id);
    }
  }

  return flattenedReplicaGroupsFromTransposePermutation(
      meshAxisNames, commAxisNames, commAxisSet, axisSizes, deviceIds,
      totalDevices);
}

}  // namespace stablehlo
}  // namespace mlir
