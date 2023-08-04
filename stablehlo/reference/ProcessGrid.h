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

#ifndef STABLEHLO_REFERENCE_PROCESSGRID_H_
#define STABLEHLO_REFERENCE_PROCESSGRID_H_

#include <condition_variable>
#include <cstdint>
#include <map>
#include <mutex>
#include <utility>

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/reference/Tensor.h"

namespace mlir {
namespace stablehlo {

// Object to represent StableHLO `process_id`.
struct ProcessId {
  /// StableHLO `replica_id`.
  uint32_t replicaId;

  /// StableHLO `partition_id`.
  uint32_t partitionId;

  bool operator<(const ProcessId &other) const {
    return std::pair<uint32_t, uint32_t>{replicaId, partitionId} <
           std::pair<uint32_t, uint32_t>{other.replicaId, other.partitionId};
  }

  bool operator==(const ProcessId &other) const {
    return std::pair<uint32_t, uint32_t>{replicaId, partitionId} ==
           std::pair<uint32_t, uint32_t>{other.replicaId, other.partitionId};
  }
};

// Object to represent StableHLO process group.
struct ProcessGroup : SmallVector<ProcessId> {};

// Object to represent StableHLO process groups.
class ProcessGroups : public SmallVector<ProcessGroup> {
 public:
  /// Returns an array of process groups.
  SmallVector<ProcessGroup> find(ProcessId &processId);
};

/// Class to model a process grid.
class ProcessGrid {
 public:
  /// \name Constructors
  /// @{
  ProcessGrid(uint32_t numReplicas, uint32_t numPartitions);
  /// @}

  /// Cross-replica communication strategy.
  ProcessGroups crossReplica(SmallVector<SmallVector<uint32_t>> replicaGroups);

  /// Cross-partition communication strategy.
  ProcessGroups crossPartition(
      SmallVector<SmallVector<uint32_t>> partitionGroups);

  /// Synchronizes running processes in the `ProcessGroup` and returns the
  /// collected resources from all partitipating processes in the process group.
  SmallVector<std::pair<ProcessId, Tensor>> rendezvous(
      ProcessGroup processGroup, int64_t channelId, ProcessId &processId,
      const Tensor &operand);

 private:
  /// The number of replicas the interpreter models.
  uint32_t numReplicas_;

  /// The number of partitions the interpreter models.
  uint32_t numPartitions_;

  /// Synchronization primitive used in `rendezvous` function.
  std::map<std::pair<ProcessGroup, int64_t>, std::mutex> resourceLockMap_;

  /// Synchronization primitive used in `rendezvous` function.
  std::map<std::pair<ProcessGroup, int64_t>, std::condition_variable>
      resourceConditionMap_;

  /// Internal mapping of StableHLO `channel_id` to its value.
  std::map<std::pair<ProcessGroup, int64_t>,
           SmallVector<std::pair<ProcessId, Tensor>>>
      channels_;
};

struct Process {
  /// StableHLO `process_id`.
  ProcessId id;

  /// StableHLO process grid.
  ProcessGrid *grid;

  /// Cross-replica communication strategy.
  ProcessGroups crossReplica(SmallVector<SmallVector<uint32_t>> replicaGroups);

  /// Cross-partition communication strategy.
  ProcessGroups crossPartition(
      SmallVector<SmallVector<uint32_t>> partitionGroups);

  /// Synchronizes running processes in the `ProcessGroup` and returns the
  /// collected resources from all partitipating processes in the process group.
  SmallVector<std::pair<ProcessId, Tensor>> rendezvous(
      ProcessGroup processGroup, int64_t channelId, const Tensor &operand);
};

}  // namespace stablehlo
}  // namespace mlir

#endif  // STABLEHLO_REFERENCE_PROCESSGRID_H_
