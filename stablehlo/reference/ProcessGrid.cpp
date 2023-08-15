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

#include "stablehlo/reference/ProcessGrid.h"

#include <condition_variable>
#include <mutex>
#include <optional>
#include <utility>

#include "stablehlo/reference/Tensor.h"

namespace mlir {
namespace stablehlo {

//===----------------------------------------------------------------------===//
// ProcessId.
//===----------------------------------------------------------------------===//

bool ProcessId::operator!=(const ProcessId &other) const {
  return !(*this == other);
}

bool ProcessId::operator<(const ProcessId &other) const {
  return std::pair<uint32_t, uint32_t>{replicaId, partitionId} <
         std::pair<uint32_t, uint32_t>{other.replicaId, other.partitionId};
}

bool ProcessId::operator==(const ProcessId &other) const {
  return std::pair<uint32_t, uint32_t>{replicaId, partitionId} ==
         std::pair<uint32_t, uint32_t>{other.replicaId, other.partitionId};
}

//===----------------------------------------------------------------------===//
// ProcessGroups.
//===----------------------------------------------------------------------===//

std::optional<ProcessGroup> ProcessGroups::findGroup(ProcessId processId) {
  for (auto processGroup : *this)
    for (auto id : processGroup)
      if (id == processId) return processGroup;

  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// RendezvousResult.
//===----------------------------------------------------------------------===//

void RendezvousResult::clear() { result_.clear(); }

void RendezvousResult::insert(ProcessId processId, Tensor tensor) {
  result_[processId] = tensor;
}

Tensor RendezvousResult::lookup(ProcessId processId) {
  auto it = result_.find(processId);
  if (it != result_.end()) return it->second;
  return {};
}

SmallVector<Tensor> RendezvousResult::getSortedTensors() {
  return llvm::to_vector(
      llvm::map_range(result_, [](const auto &pair) { return pair.second; }));
}

size_t RendezvousResult::size() { return result_.size(); }

//===----------------------------------------------------------------------===//
// ProcessGrid.
//===----------------------------------------------------------------------===//

ProcessGrid::ProcessGrid(uint32_t numReplicas, uint32_t numPartitions)
    : numReplicas_(numReplicas), numPartitions_(numPartitions) {}

ProcessGroups ProcessGrid::crossPartition(
    SmallVector<SmallVector<uint32_t>> partitionGroups) {
  ProcessGroups processGroups;
  for (const auto &partitionGroup : partitionGroups) {
    for (uint32_t replicaId = 0; replicaId < numReplicas_; ++replicaId) {
      ProcessGroup processGroup;
      for (uint32_t partitionId : partitionGroup)
        processGroup.push_back({replicaId, partitionId});
      processGroups.push_back(processGroup);
    }
  }
  return processGroups;
}

ProcessGroups ProcessGrid::crossReplica(
    SmallVector<SmallVector<uint32_t>> replicaGroups) {
  ProcessGroups processGroups;
  for (const auto &replicaGroup : replicaGroups) {
    for (uint32_t partitionId = 0; partitionId < numPartitions_;
         ++partitionId) {
      ProcessGroup processGroup;
      for (uint32_t replicaId : replicaGroup)
        processGroup.push_back({replicaId, partitionId});
      processGroups.push_back(processGroup);
    }
  }
  return processGroups;
}

ProcessGroups ProcessGrid::crossReplicaAndPartition(
    SmallVector<SmallVector<uint32_t>> replicaGroups) {
  ProcessGroups processGroups;
  for (const auto &replicaGroup : replicaGroups) {
    ProcessGroup processGroup;
    for (uint32_t partitionId = 0; partitionId < numPartitions_; ++partitionId)
      for (uint32_t replicaId : replicaGroup)
        processGroup.push_back({replicaId, partitionId});
    processGroups.push_back(processGroup);
  }
  return processGroups;
}

ProcessGroups ProcessGrid::flattenedIds(
    SmallVector<SmallVector<uint32_t>> flattenedIdGroups) {
  ProcessGroups processGroups;
  for (const auto &flattenedIdGroup : flattenedIdGroups) {
    ProcessGroup processGroup;
    for (auto flattenedId : flattenedIdGroup) {
      uint32_t replicaId = flattenedId / numPartitions_;
      uint32_t partitionId = flattenedId % numPartitions_;
      processGroup.push_back({replicaId, partitionId});
    }
    processGroups.push_back(processGroup);
  }
  return processGroups;
}

std::mutex &ProcessGrid::getRendezvousLock(ProcessGroup processGroup,
                                           ChannelId channelId) {
  std::lock_guard<std::mutex> lock(rendezvousLock_);
  std::pair<ProcessGroup, ChannelId> channelKey(processGroup, channelId);
  return channelLocks_[channelKey];
}

void ProcessGrid::outfeed(ArrayRef<Tensor> inputs) {
  std::lock_guard<std::mutex> lock(outfeedLock_);
  outfeed_.emplace(inputs);
}

RendezvousResult ProcessGrid::rendezvous(ProcessGroup processGroup,
                                         ChannelId channelId,
                                         ProcessId processId,
                                         const Tensor &operand) {
  std::pair<ProcessGroup, ChannelId> channelKey(processGroup, channelId);
  {
    std::lock_guard<std::mutex> lock(
        getRendezvousLock(processGroup, channelId));
    if (channels_[channelKey].size() == processGroup.size())
      channels_[channelKey].clear();

    channels_[channelKey].insert(processId, operand);
  }
  {
    std::unique_lock<std::mutex> lock(
        getRendezvousLock(processGroup, channelId));
    if (channels_[channelKey].size() == processGroup.size())
      channelConditions_[channelKey].notify_all();

    if (!channelConditions_[channelKey].wait_for(
            lock, std::chrono::seconds(3), [&] {
              return channels_[channelKey].size() == processGroup.size();
            }))
      llvm::report_fatal_error("rendezvous timed out");

    return channels_[channelKey];
  }
}

}  // namespace stablehlo
}  // namespace mlir
