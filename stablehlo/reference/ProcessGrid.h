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

#ifndef STABLEHLO_REFERENCE_PROCESSGRID_H
#define STABLEHLO_REFERENCE_PROCESSGRID_H

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

using ChannelId = int64_t;

// StableHLO `process_id`.
struct ProcessId {
  /// StableHLO `replica_id`.
  uint32_t replicaId;

  /// StableHLO `partition_id`.
  uint32_t partitionId;

  bool operator<(const ProcessId &other) const;

  bool operator==(const ProcessId &other) const;

  bool operator!=(const ProcessId &other) const;
};

// StableHLO `process_group`.
struct ProcessGroup : public SmallVector<ProcessId> {};

// StableHLO `process_groups`.
struct ProcessGroups : public SmallVector<ProcessGroup> {};

class RendezvousResult {
 public:
  /// Iterates through the map `result_` and returns the value associated with
  /// the key `processId`.
  /// If key is not found, return an empty `Tensor`.
  Tensor lookup(ProcessId processId);

  /// Inserts `tensor` into the map `result_` using the key `processId`.
  void insert(ProcessId processId, Tensor tensor);

  /// Erases all elements in the map `result_`.
  void clear();

  /// Returns the size of `result_`.
  size_t size();

 private:
  ///
  std::map<ProcessId, Tensor> result_;
};

/// Class to model a process grid.
class ProcessGrid {
 public:
  /// \name Constructors
  /// @{
  ProcessGrid(uint32_t numReplicas, uint32_t numPartitions);
  /// @}

  /// StableHLO `cross_replcia`.
  ProcessGroups crossReplica(SmallVector<SmallVector<uint32_t>> replicaGroups);

  /// StableHLO `cross_partition`.
  ProcessGroups crossPartition(
      SmallVector<SmallVector<uint32_t>> partitionGroups);

  /// Each participating process in the `processGroup` appends its data
  /// `operand` to the `channels_` map using the pair (processGroup, channelId)
  /// as a key. `channels_` is cleared before any process adds its data.
  /// `rendezvous` then returns `RendezvousResult` containing a mapping of
  /// `processId` to its data `operand` once all participating processes have
  /// successfully added their data. Throws an error after a timeout when
  /// synchronization deadlocks.
  RendezvousResult rendezvous(ProcessGroup processGroup, int64_t channelId,
                              ProcessId processId, const Tensor &operand);

 private:
  /// StableHLO `num_replicas`.
  uint32_t numReplicas_;

  /// StableHLO `num_partitions`.
  uint32_t numPartitions_;

  /// Internal storage used to implement `rendezvous`.
  /// Each call to `rendezvous`, i.e. each combination `processGroup` and
  /// `channelId`, has its own key in the map.
  /// Within the implementation of `rendezvous`, the value corresponding to
  /// this key is gradually populated with tensors arriving from different
  /// processes in the process group.
  std::map<std::pair<ProcessGroup, ChannelId>, RendezvousResult> channels_;

  /// Synchronization primitive used to manage concurrent access to `channels_`.
  std::map<std::pair<ProcessGroup, ChannelId>, std::mutex> channelLocks_;

  /// Synchronization primitive used to manage concurrent access to `channels_`.
  std::map<std::pair<ProcessGroup, ChannelId>, std::condition_variable>
      channelConditions_;
};

}  // namespace stablehlo
}  // namespace mlir

#endif  // STABLEHLO_REFERENCE_PROCESSGRID_H
