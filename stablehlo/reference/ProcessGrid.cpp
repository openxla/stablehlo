/* Copyright 2023-2024 The StableHLO Authors.

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

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <map>
#include <mutex>
#include <optional>
#include <queue>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Support/LLVM.h"
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
    if (llvm::find(processGroup, processId) != processGroup.end())
      return processGroup;

  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// RendezvousResult.
//===----------------------------------------------------------------------===//

RendezvousResult::RendezvousResult(
    std::map<ProcessId, SmallVector<Tensor>> const &results)
    : results_(results) {}

void RendezvousResult::insert(ProcessId processId,
                              SmallVector<Tensor> tensors) {
  results_[processId] = tensors;
}

SmallVector<Tensor> RendezvousResult::lookup(ProcessId processId) const {
  auto it = results_.find(processId);
  if (it != results_.end()) return it->second;
  return {};
}

SmallVector<SmallVector<Tensor>> RendezvousResult::getSortedTensors() const {
  return llvm::map_to_vector(results_,
                             [](const auto &pair) { return pair.second; });
}

bool RendezvousResult::hasMatchingOperandsCount() const {
  auto count = results_.begin()->second.size();
  for (const auto &it : results_)
    if (count != it.second.size()) return false;
  return true;
}

//===----------------------------------------------------------------------===//
// ThreadSafeMap.
//===----------------------------------------------------------------------===//

template <typename K, typename V>
V &detail::ThreadSafeMap<K, V>::operator[](const K &key) {
  std::lock_guard<std::mutex> lock(lock_);
  return map_[key];
}

//===----------------------------------------------------------------------===//
// ThreadSafeSet.
//===----------------------------------------------------------------------===//

template <typename T>
bool detail::ThreadSafeSet<T>::contains(T value) {
  std::lock_guard<std::mutex> lock(lock_);
  return set_.find(value) != set_.end();
}

template <typename T>
void detail::ThreadSafeSet<T>::erase(T value) {
  std::lock_guard<std::mutex> lock(lock_);
  set_.erase(value);
}

template <typename T>
void detail::ThreadSafeSet<T>::insert(T value) {
  std::lock_guard<std::mutex> lock(lock_);
  set_.insert(value);
}

//===----------------------------------------------------------------------===//
// ThreadSafeQueue.
//===----------------------------------------------------------------------===//

template <typename T>
detail::ThreadSafeQueue<T>::ThreadSafeQueue(const std::queue<T> &queue)
    : queue_(queue) {}

template <typename T>
T detail::ThreadSafeQueue<T>::pop() {
  std::lock_guard<std::mutex> lock(lock_);
  auto result = queue_.front();
  queue_.pop();
  return result;
}

template <typename T>
void detail::ThreadSafeQueue<T>::push(T input) {
  std::lock_guard<std::mutex> lock(lock_);
  queue_.emplace(input);
}

//===----------------------------------------------------------------------===//
// ProcessGrid.
//===----------------------------------------------------------------------===//

ProcessGrid::ProcessGrid(uint32_t numReplicas, uint32_t numPartitions,
                         std::queue<StringAttr> &infeed)
    : numReplicas_(numReplicas),
      numPartitions_(numPartitions),
      infeed_(infeed) {}

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

StringAttr ProcessGrid::infeed() { return infeed_.pop(); }

void ProcessGrid::outfeed(ArrayRef<Tensor> inputs) {
  outfeed_.push(llvm::to_vector(inputs));
}

SmallVector<Tensor> ProcessGrid::recv(ChannelId channelId,
                                      ProcessId processId) {
  std::unique_lock<std::mutex> lock(sendRecvChannels_[channelId].mutex);
  sendRecvReady_.insert(channelId);
  sendRecvConditions_[channelId].notify_one();

  if (!sendRecvConditions_[channelId].wait_for(
          lock, std::chrono::seconds(3),
          [&] { return !sendRecvChannels_[channelId].result.empty(); }))
    llvm::report_fatal_error("recv timed out");

  auto result = sendRecvChannels_[channelId].result;
  sendRecvChannels_[channelId].result.clear();
  return result;
}

RendezvousResult ProcessGrid::rendezvous(ProcessGroup processGroup,
                                         ChannelId channelId,
                                         ProcessId processId,
                                         ArrayRef<Tensor> operands) {
  // Process wait/notify logic below doesn't work for single process.
  if (processGroup.size() == 1) {
    std::map<ProcessId, SmallVector<Tensor>> results;
    results[processId] = SmallVector<Tensor>(operands);
    return RendezvousResult(results);
  }

  std::pair<ProcessGroup, ChannelId> channelKey(processGroup, channelId);
  auto &state = channels_[channelKey];

  std::unique_lock<std::mutex> lock(state.mutex);
  state.values[processId] = SmallVector<Tensor>(operands);
  state.useCount++;

  // After each process contributes, wait for the last process to notify.
  if (state.values.size() < processGroup.size()) {
    if (!channelConditions_[channelKey].wait_for(
            lock, std::chrono::seconds(3),
            [&] { return state.values.size() == processGroup.size(); }))
      llvm::report_fatal_error("rendezvous timed out");
  } else {
    state.result = std::move(state.values);
    channelConditions_[channelKey].notify_all();
  }

  state.useCount--;

  if (!state.result.hasMatchingOperandsCount())
    llvm::report_fatal_error("Mismatched number of operands per process");

  return state.useCount > 0 ? state.result : std::move(state.result);
}

void ProcessGrid::send(ArrayRef<Tensor> inputs, ChannelId channelId,
                       ProcessId processId) {
  std::unique_lock<std::mutex> lock(sendRecvChannels_[channelId].mutex);
  if (!sendRecvConditions_[channelId].wait_for(
          lock, std::chrono::seconds(3),
          [&] { return sendRecvReady_.contains(channelId); }))
    llvm::report_fatal_error("send timed out");

  sendRecvChannels_[channelId].result = llvm::to_vector(inputs);
  sendRecvReady_.erase(channelId);
  sendRecvConditions_[channelId].notify_one();
}

}  // namespace stablehlo
}  // namespace mlir
