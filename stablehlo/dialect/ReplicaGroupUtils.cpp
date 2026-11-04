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
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {
namespace stablehlo {

namespace {

struct ReindexedAxes {
  SmallVector<int64_t> splitAxisSizes;
  SmallVector<int64_t> groupedAxisIndices;
};

// Generates replica groups from the reshaped mesh axis sizes and the indices of
// the communication axes using a Reshape-Transpose permutation.
SmallVector<SmallVector<int64_t>>
flattenedReplicaGroupsFromTransposePermutation(
    ArrayRef<int64_t> axisSizes, ArrayRef<int64_t> groupedAxisIndices,
    ArrayRef<int64_t> deviceIds, int64_t totalDevices) {
  llvm::DenseSet<int64_t> groupedAxisSet(groupedAxisIndices.begin(),
                                         groupedAxisIndices.end());
  SmallVector<int64_t> transposeAxes;
  // Non-grouped axes first
  for (size_t i = 0; i < axisSizes.size(); ++i) {
    if (!groupedAxisSet.count(i)) {
      transposeAxes.push_back(i);
    }
  }
  // Grouped axes in the specified order
  for (int64_t idx : groupedAxisIndices) {
    transposeAxes.push_back(idx);
  }

  SmallVector<int64_t> transposedSizes(axisSizes.size());
  for (size_t i = 0; i < axisSizes.size(); ++i) {
    transposedSizes[i] = axisSizes[transposeAxes[i]];
  }

  // Compute strides for reshaped shape
  SmallVector<int64_t> originalStrides(axisSizes.size(), 1);
  for (int i = static_cast<int>(axisSizes.size()) - 2; i >= 0; --i) {
    originalStrides[i] = originalStrides[i + 1] * axisSizes[i + 1];
  }

  // Compute strides for transposed shape
  SmallVector<int64_t> transposedStrides(axisSizes.size(), 1);
  for (int i = static_cast<int>(axisSizes.size()) - 2; i >= 0; --i) {
    transposedStrides[i] = transposedStrides[i + 1] * transposedSizes[i + 1];
  }

  int64_t numDevicesPerGroup = 1;
  for (int64_t idx : groupedAxisIndices) {
    numDevicesPerGroup *= axisSizes[idx];
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
      for (size_t k = 0; k < axisSizes.size(); ++k) {
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

// Splits mesh axes based on sub-axis references and computes the corresponding
// indices for the communication axes.
FailureOr<ReindexedAxes> computeReindexedAxes(ArrayRef<MeshAxisAttr> axesInMesh,
                                              ArrayAttr commAxes,
                                              Location loc) {
  ReindexedAxes result;

  // Validate commAxes and verify that all mesh axes exist and have valid sizes.
  for (auto attr : commAxes) {
    auto shloAxisRef = llvm::dyn_cast<AxisRefAttr>(attr);
    if (!shloAxisRef) {
      return emitError(loc) << "expected AxisRefAttr in comm_axes";
    }
    StringRef axisName = shloAxisRef.getName();
    bool found = false;
    for (auto meshAxis : axesInMesh) {
      if (meshAxis.getName() == axisName) {
        found = true;
        if (auto subAxisInfo = shloAxisRef.getSubAxisInfo()) {
          int64_t preSize = subAxisInfo.getPreSize();
          int64_t size = subAxisInfo.getSize();
          if (preSize < 1 || size < 1) {
            return emitError(loc)
                   << "sub-axis pre_size and size must be at least 1";
          }
          int64_t nextPreSize = preSize * size;
          if (nextPreSize > meshAxis.getSize() ||
              meshAxis.getSize() % nextPreSize != 0) {
            return emitError(loc)
                   << "sub-axis (pre_size * size) must divide mesh axis size";
          }
        }
        break;
      }
    }
    if (!found) {
      return emitError(loc)
             << "axis '" << axisName << "' not found in mesh definition";
    }
  }

  // Split each mesh axis according to the referenced subaxes.
  struct SplitDim {
    StringRef axisName;
    int64_t preSize;
    int64_t size;
    int64_t dimIndex;
  };
  SmallVector<SplitDim> splitDims;

  for (auto meshAxis : axesInMesh) {
    StringRef axisName = meshAxis.getName();
    int64_t axisSize = meshAxis.getSize();

    if (axisSize == 1) {
      int64_t dimIdx = result.splitAxisSizes.size();
      result.splitAxisSizes.push_back(1);
      splitDims.push_back({axisName, /*preSize=*/1, /*size=*/1, dimIdx});
      continue;
    }

    SmallVector<int64_t> preSizes = {1, axisSize};
    for (auto attr : commAxes) {
      auto shloAxisRef = llvm::cast<AxisRefAttr>(attr);
      if (shloAxisRef.getName() == axisName) {
        if (auto subAxisInfo = shloAxisRef.getSubAxisInfo()) {
          preSizes.push_back(subAxisInfo.getPreSize());
          preSizes.push_back(subAxisInfo.getPreSize() * subAxisInfo.getSize());
        }
      }
    }

    llvm::sort(preSizes);
    preSizes.erase(llvm::unique(preSizes), preSizes.end());

    for (size_t j = 0; j < preSizes.size() - 1; ++j) {
      int64_t segPreSize = preSizes[j];
      int64_t segSize = preSizes[j + 1] / segPreSize;
      int64_t dimIdx = result.splitAxisSizes.size();
      result.splitAxisSizes.push_back(segSize);
      splitDims.push_back({axisName, segPreSize, segSize, dimIdx});
    }
  }

  // Map each communication axis to its corresponding split dimension.
  llvm::DenseSet<int64_t> groupedSet;
  for (auto attr : commAxes) {
    auto shloAxisRef = llvm::cast<AxisRefAttr>(attr);
    StringRef axisName = shloAxisRef.getName();
    int64_t reqPreSize = 1;
    int64_t reqSize = 0;
    if (auto subAxisInfo = shloAxisRef.getSubAxisInfo()) {
      reqPreSize = subAxisInfo.getPreSize();
      reqSize = subAxisInfo.getSize();
    } else {
      for (auto meshAxis : axesInMesh) {
        if (meshAxis.getName() == axisName) {
          reqSize = meshAxis.getSize();
          break;
        }
      }
    }

    bool matched = false;
    for (const auto& splitDim : splitDims) {
      if (splitDim.axisName == axisName && splitDim.preSize == reqPreSize &&
          splitDim.size == reqSize) {
        if (!groupedSet.insert(splitDim.dimIndex).second) {
          return emitError(loc)
                 << "Duplicate or overlapping communication axis: " << axisName;
        }
        result.groupedAxisIndices.push_back(splitDim.dimIndex);
        matched = true;
        break;
      }
    }
    if (!matched) {
      return emitError(loc) << "Invalid or overlapping communication axis on '"
                            << axisName << "'";
    }
  }

  return result;
}

}  // namespace

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

  FailureOr<ReindexedAxes> reindexedAxes =
      computeReindexedAxes(mesh.getAxes(), commAxes, loc);
  if (failed(reindexedAxes)) return failure();

  int64_t totalDevices = 1;
  for (auto meshAxis : mesh.getAxes()) {
    totalDevices *= llvm::cast<stablehlo::MeshAxisAttr>(meshAxis).getSize();
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
      reindexedAxes->splitAxisSizes, reindexedAxes->groupedAxisIndices,
      deviceIds, totalDevices);
}

}  // namespace stablehlo
}  // namespace mlir
