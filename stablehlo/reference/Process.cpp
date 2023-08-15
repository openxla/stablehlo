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

#include "stablehlo/reference/Process.h"

namespace mlir {
namespace stablehlo {

Process::Process(ProcessId id, ProcessGrid *grid) : id_(id), grid_(grid) {}

ProcessGroups Process::crossPartition(
    SmallVector<SmallVector<uint32_t>> partitionGroups) {
  return grid_->crossPartition(partitionGroups);
}

ProcessGroups Process::crossReplica(
    SmallVector<SmallVector<uint32_t>> replicaGroups) {
  return grid_->crossReplica(replicaGroups);
}

ProcessGroups Process::crossReplicaAndPartition(
    SmallVector<SmallVector<uint32_t>> replicaGroups) {
  return grid_->crossReplicaAndPartition(replicaGroups);
}

ProcessGroups Process::flattenedIds(
    SmallVector<SmallVector<uint32_t>> flattenedIdGroups) {
  return grid_->flattenedIds(flattenedIdGroups);
}

ProcessId Process::getId() { return id_; }

void Process::outfeed(ArrayRef<Tensor> inputs) { grid_->outfeed(inputs); }

RendezvousResult Process::rendezvous(ProcessGroup processGroup,
                                     ChannelId channelId,
                                     const Tensor &operand) {
  return grid_->rendezvous(processGroup, channelId, getId(), operand);
}

}  // namespace stablehlo
}  // namespace mlir
