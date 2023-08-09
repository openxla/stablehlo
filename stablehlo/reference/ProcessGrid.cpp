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
#include <utility>

#include "llvm/Support/ErrorHandling.h"
#include "stablehlo/reference/Tensor.h"

namespace mlir {
namespace stablehlo {

bool ProcessId::operator<(const ProcessId &other) const {
  return std::pair<uint32_t, uint32_t>{replicaId, partitionId} <
         std::pair<uint32_t, uint32_t>{other.replicaId, other.partitionId};
}

bool ProcessId::operator==(const ProcessId &other) const {
  return std::pair<uint32_t, uint32_t>{replicaId, partitionId} ==
         std::pair<uint32_t, uint32_t>{other.replicaId, other.partitionId};
}

bool ProcessId::operator!=(const ProcessId &other) const {
  return !(*this == other);
}

Tensor RendezvousResult::lookup(ProcessId processId) {
  auto it = result_.find(processId);
  if (it != result_.end()) return it->second;
  return {};
}

void RendezvousResult::insert(ProcessId processId, Tensor tensor) {
  result_[processId] = tensor;
}

void RendezvousResult::clear() { result_.clear(); }

size_t RendezvousResult::size() { return result_.size(); }

ProcessGrid::ProcessGrid(uint32_t numReplicas, uint32_t numPartitions)
    : numReplicas_(numReplicas), numPartitions_(numPartitions) {}

ProcessGroups ProcessGrid::crossReplica(
    SmallVector<SmallVector<uint32_t>> replicaGroups) {
  ProcessGroups processGroups;
  for (auto replicaGroup : replicaGroups) {
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

ProcessGroups ProcessGrid::crossPartition(
    SmallVector<SmallVector<uint32_t>> partitionGroups) {
  ProcessGroups processGroups;
  for (auto partitionGroup : partitionGroups) {
    for (uint32_t replicaId = 0; replicaId < numReplicas_; ++replicaId) {
      ProcessGroup processGroup;
      for (uint32_t partitionId : partitionGroup)
        processGroup.push_back({replicaId, partitionId});
      processGroups.push_back(processGroup);
    }
  }
  return processGroups;
}

void ProcessGrid::outfeed(ArrayRef<Tensor> inputs) {
  outfeed_.push(llvm::to_vector(inputs));
}

RendezvousResult ProcessGrid::rendezvous(ProcessGroup processGroup,
                                         int64_t channelId, ProcessId processId,
                                         const Tensor &operand) {
  std::pair<ProcessGroup, ChannelId> channelKey(processGroup, channelId);
  {
    std::lock_guard<std::mutex> lock(channelLocks_[channelKey]);
    if (channels_[channelKey].size() == processGroup.size())
      channels_[channelKey].clear();
    channels_[channelKey].insert(processId, operand);
  }
  {
    std::unique_lock<std::mutex> lock(channelLocks_[channelKey]);
    if (channels_[channelKey].size() == processGroup.size())
      channelConditions_[channelKey].notify_all();

    if (!channelConditions_[channelKey].wait_for(
            lock, std::chrono::seconds(3), [&] {
              return channels_[channelKey].size() == processGroup.size();
            }))
      llvm::report_fatal_error("rendezvous timed out");
  }

  return channels_[channelKey];
}

}  // namespace stablehlo
}  // namespace mlir
