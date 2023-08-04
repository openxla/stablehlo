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

SmallVector<ProcessGroup> ProcessGroups::find(ProcessId &processId) {
  SmallVector<ProcessGroup> groupsFound{};

  for (auto processGroup : *this)
    for (auto id : processGroup)
      if (id == processId) groupsFound.push_back(processGroup);

  return groupsFound;
}

ProcessGroups Process::crossReplica(
    SmallVector<SmallVector<uint32_t>> replicaGroups) {
  return grid->crossReplica(replicaGroups);
}

ProcessGroups Process::crossPartition(
    SmallVector<SmallVector<uint32_t>> partitionGroups) {
  return grid->crossPartition(partitionGroups);
}

SmallVector<std::pair<ProcessId, Tensor>> Process::rendezvous(
    ProcessGroup processGroup, int64_t channelId, const Tensor &operand) {
  return grid->rendezvous(processGroup, channelId, id, operand);
}

ProcessGrid::ProcessGrid(uint32_t numReplicas, uint32_t numPartitions)
    : numReplicas_(numReplicas),
      numPartitions_(numPartitions),
      resourceLockMap_(
          std::map<std::pair<ProcessGroup, int64_t>, std::mutex>()),
      resourceConditionMap_(std::map<std::pair<ProcessGroup, int64_t>,
                                     std::condition_variable>()),
      channels_(std::map<std::pair<ProcessGroup, int64_t>,
                         SmallVector<std::pair<ProcessId, Tensor>>>()) {}

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

SmallVector<std::pair<ProcessId, Tensor>> ProcessGrid::rendezvous(
    ProcessGroup processGroup, int64_t channelId, ProcessId &processId,
    const Tensor &operand) {
  {
    std::unique_lock<std::mutex> lk(
        resourceLockMap_[{processGroup, channelId}]);
    channels_[{processGroup, channelId}].push_back({processId, operand});
  }

  resourceConditionMap_[{processGroup, channelId}].notify_all();

  {
    std::unique_lock<std::mutex> lk(
        resourceLockMap_[{processGroup, channelId}]);
    if (channels_[{processGroup, channelId}].size() == processGroup.size())
      resourceConditionMap_[{processGroup, channelId}].notify_all();

    if (!resourceConditionMap_[{processGroup, channelId}].wait_for(
            lk, std::chrono::seconds(3), [&] {
              return channels_[{processGroup, channelId}].size() ==
                     processGroup.size();
            }))
      llvm::report_fatal_error("rendezvous timed out");

    std::sort(channels_[{processGroup, channelId}].begin(),
              channels_[{processGroup, channelId}].end(),
              [](std::pair<ProcessId, Tensor> &lhs,
                 std::pair<ProcessId, Tensor> &rhs) {
                return lhs.first < rhs.first;
              });
  }

  return channels_[{processGroup, channelId}];
}

}  // namespace stablehlo
}  // namespace mlir
