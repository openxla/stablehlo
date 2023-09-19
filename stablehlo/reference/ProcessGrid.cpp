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

RendezvousResult::RendezvousResult(std::map<ProcessId, Tensor> result)
    : result_(result) {}

void RendezvousResult::insert(ProcessId processId, Tensor tensor) {
  result_[processId] = tensor;
}

Tensor RendezvousResult::lookup(ProcessId processId) const {
  auto it = result_.find(processId);
  if (it != result_.end()) return it->second;
  return {};
}

SmallVector<Tensor> RendezvousResult::getSortedTensors() const {
  return llvm::to_vector(
      llvm::map_range(result_, [](const auto &pair) { return pair.second; }));
}

//===----------------------------------------------------------------------===//
// ThreadSafeMap.
//===----------------------------------------------------------------------===//

template <typename K, typename V>
V &ProcessGrid::ThreadSafeMap<K, V>::operator[](const K &key) {
  std::lock_guard<std::mutex> lock(lock_);
  return map_[key];
}

//===----------------------------------------------------------------------===//
// ThreadSafeQueue.
//===----------------------------------------------------------------------===//

void ProcessGrid::ThreadSafeQueue::push(ArrayRef<Tensor> inputs) {
  std::lock_guard<std::mutex> lock(lock_);
  queue_.emplace(inputs);
}

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

void ProcessGrid::outfeed(ArrayRef<Tensor> inputs) { outfeed_.push(inputs); }

std::shared_ptr<RendezvousResult const> ProcessGrid::rendezvous(
    ProcessGroup processGroup, ChannelId channelId, ProcessId processId,
    const Tensor &operand) {
  std::pair<ProcessGroup, ChannelId> channelKey(processGroup, channelId);
  // Process wait/notify logic below doesn't work for single process.
  if (processGroup.size() == 1)
    return std::make_shared<RendezvousResult>(
        RendezvousResult({std::pair{processId, operand}}));

  auto &state = channels_[channelKey];

  std::unique_lock<std::mutex> lock(state.mutex);
  state.values[processId] = operand;

  if (state.values.size() == processGroup.size()) {
    // If values are full, that means all other processes are currently waiting.
    // The last process to contribute moves the values into the result
    // then waits for each process to return a copy of the result before
    // cleaning up the state variable for future computations in this process
    // grid.
    state.result = std::make_shared<RendezvousResult>(state.values);
    state.values.clear();
    channelConditions_[channelKey].notify_one();

    // The last process to contribute waits until the rest of the processes have
    // read the values.
    if (!channelConditions_[channelKey].wait_for(
            lock, std::chrono::seconds(3), [&] {
              return state.result.use_count() >=
                     static_cast<int64_t>(processGroup.size());
            }))
      llvm::report_fatal_error(
          "rendezvous timed out: not all processes have contributed yet");

    if (state.result.use_count() > static_cast<int64_t>(processGroup.size()))
      llvm::report_fatal_error(
          "Each process should have only one shared access to the result.");

    // The last process to contribute takes the result from the state to allow
    // the process that contributed last to exit the function.
    channelConditions_[channelKey].notify_one();
    return std::move(state.result);
  }

  // Wait for all processes to contribute values.
  if (!channelConditions_[channelKey].wait_for(
          lock, std::chrono::seconds(3),
          [&] { return state.result != nullptr; }))
    llvm::report_fatal_error(
        "rendezvous timed out: not all process has received the results yet");

  // Copy result from the state before notifying.
  auto result = state.result;
  channelConditions_[channelKey].notify_one();

  // Wait for the remaining processes to have retrieved the result. In other
  // words, wait until the last process to contribute exit the function.
  if (!channelConditions_[channelKey].wait_for(
          lock, std::chrono::seconds(3),
          [&] { return state.result == nullptr; }))
    llvm::report_fatal_error(
        "rendezvous timed out: not all process has received the results yet");

  return result;
}

}  // namespace stablehlo
}  // namespace mlir
