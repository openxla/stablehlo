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
#include <optional>
#include <queue>
#include <utility>

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

  bool operator!=(const ProcessId &other) const;

  // The sort order for ProcessId is not defined in StableHLO, and it's
  // internally used in ProcessGrid::rendezvous as part of a sorted key on the
  // map. This operator is conveniently used to help define the ordering since
  // ordering is defined for StableHLO process group.
  bool operator<(const ProcessId &other) const;

  bool operator==(const ProcessId &other) const;
};

// StableHLO `process_group`.
class ProcessGroup : public SmallVector<ProcessId> {};

// StableHLO `process_groups`.
class ProcessGroups : public SmallVector<ProcessGroup> {
 public:
  /// Iterates through the ProcessGroups and finds the first ProcessGroup
  /// containing the `processId`. If the group is not found, std::nullopt is
  /// returned.
  std::optional<ProcessGroup> findGroup(ProcessId processId);
};

/// Represents a result of a `ProcessGrid::rendezvous` where multiple processes
/// synchronize at a barrier and contribute a Tensor each.
/// This class is pretty much a map from ProcessId to Tensor, with the
/// map-like API.
class RendezvousResult {
 public:
  /// Erases all elements in the map.
  void clear();

  /// Iterates through the (ProcessId, Tensor) map entires and returns a vector
  /// of Tensors sorted by ProcessId--(replicaId, partitionId) pair--in
  /// lexicographical order.
  SmallVector<Tensor> getSortedTensors();

  /// Inserts `tensor` into the map using the key `processId`.
  void insert(ProcessId processId, Tensor tensor);

  /// Iterates through the map and returns the value associated with the key
  /// `processId`. If key is not found, return an empty `Tensor`.
  Tensor lookup(ProcessId processId);

  /// Returns the number of elements in the map.
  size_t size();

 private:
  /// Internal map representation of the result of `ProcessGrid::rendezvous`.
  std::map<ProcessId, Tensor> result_;
};

/// StableHLO process grid.
class ProcessGrid {
 public:
  /// \name Constructors
  /// @{
  ProcessGrid(uint32_t numReplicas, uint32_t numPartitions);
  /// @}

  /// StableHLO `cross_partition` communication strategy.
  ProcessGroups crossPartition(
      SmallVector<SmallVector<uint32_t>> partitionGroups);

  /// StableHLO `cross_replica` communication strategy.
  ProcessGroups crossReplica(SmallVector<SmallVector<uint32_t>> replicaGroups);

  /// StableHLO `cross_replica_and_partition` communication strategy.
  ProcessGroups crossReplicaAndPartition(
      SmallVector<SmallVector<uint32_t>> replicaGroups);

  /// StableHLO `flattened_ids` communication strategy.
  ProcessGroups flattenedIds(
      SmallVector<SmallVector<uint32_t>> flattenedIdGroups);

  /// Inserts `inputs` to StableHLO `outfeed`.
  void outfeed(ArrayRef<Tensor> inputs);

  /// Synchronize a StableHLO process with the `processId` with other StableHLO
  /// processes in the `processGroup` using a `channelId`.
  ///
  /// A call to this method represents a barrier, i.e. it blocks the calling
  /// OS thread until all StableHLO processes from the `processGroup` call this
  /// method with the same `channelId`. If the calling OS thread doesn't
  /// correspond to the StableHLO process with `processId`, the behavior is
  /// undefined.
  ///
  /// If any of the StableHLO processes from `processGroup` fail to arrive
  /// at the barrier within 3 seconds, the `rendezvous` fails with a fatal
  /// error for all calling OS threads. This is to make sure that errors in
  /// underlying StableHLO programs or bugs in the StableHLO interpreter don't
  /// deadlock the interpreter.
  ///
  /// At the barrier, each StableHLO process contributes a tensor, and these
  /// tensors are accumulated in `RendezvousResult` which is returned to all
  /// callers once the barrier has been reached by all StableHLO processes.
  RendezvousResult rendezvous(ProcessGroup processGroup, ChannelId channelId,
                              ProcessId processId, const Tensor &operand);

 private:
  /// Obtain a mutex that is shared between all processes participating in
  /// a call to `rendezvous` for a given combination of `processGroup` and
  /// `channelId`.
  std::mutex &getRendezvousLock(ProcessGroup processGroup, ChannelId channelId);

  /// StableHLO `num_replicas`.
  const uint32_t numReplicas_;

  /// StableHLO `num_partitions`.
  const uint32_t numPartitions_;

  /// StableHLO `outfeed` represented as a queue.
  std::queue<SmallVector<Tensor>> outfeed_;

  std::mutex outfeedLock_;

  /// Synchronization primitive used to manage concurrent access to
  /// `channelLocks_`.
  std::mutex rendezvousLock_;

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
