/* Copyright 2022 The StableHLO Authors.

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

#include "stablehlo/reference/Index.h"

namespace mlir {
namespace stablehlo {

const Index &IndexSpaceIterator::operator*() const {
  if (!index_)
    llvm::report_fatal_error("Dereferencing a past-the-end iterator.");
  return *index_;
}

const Index *IndexSpaceIterator::operator->() const { return &(*index_); }

IndexSpaceIterator &IndexSpaceIterator::operator++() {
  if (!index_)
    llvm::report_fatal_error("Incrementing a past-the-end iterator.");

  if (shape_.empty()) index_.reset();

  for (int64_t i = shape_.size() - 1; i >= 0; --i) {
    (*index_)[i] += 1;
    if ((*index_)[i] >= shape_[i]) {
      (*index_)[i] = 0;
      if (i == 0) {
        index_.reset();
        break;
      }
    } else {
      break;
    }
  }

  return *this;
}

IndexSpaceIterator IndexSpaceIterator::operator++(int) {
  IndexSpaceIterator tempIter = *this;
  ++*this;
  return tempIter;
}

}  // namespace stablehlo
}  // namespace mlir
