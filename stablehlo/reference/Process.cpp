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

#include "stablehlo/reference/Process.h"

#include <cstdint>

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/reference/ProcessGrid.h"
#include "stablehlo/reference/Tensor.h"

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

StringAttr Process::infeed() { return grid_->infeed(); }

ProcessId Process::getId() { return id_; }

void Process::outfeed(ArrayRef<Tensor> inputs) { grid_->outfeed(inputs); }

SmallVector<Tensor> Process::recv(ChannelId channelId) {
  return grid_->recv(channelId, getId());
}

RendezvousResult Process::rendezvous(ProcessGroup processGroup,
                                     ChannelId channelId,
                                     ArrayRef<Tensor> operands) {
  return grid_->rendezvous(processGroup, channelId, getId(), operands);
}

void Process::send(ArrayRef<Tensor> inputs, ChannelId channelId) {
  grid_->send(inputs, channelId, getId());
}

}  // namespace stablehlo
}  // namespace mlir
