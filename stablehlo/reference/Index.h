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

#ifndef STABLEHLO_REFERENCE_INDEX_H_
#define STABLEHLO_REFERENCE_INDEX_H_

#include <cstdint>
#include <vector>

#include "llvm/ADT/ArrayRef.h"

namespace mlir {
namespace stablehlo {

/// Iterator over the indices of a tensor in major-to-minor order. As an
/// example, for a tensor with shape [2,3], the iterator enumerates the
/// indices (0,0), (0,1), (0,2), (1,0), (1,1), and (1,2).
class IndexSpaceIterator {
 public:
  /// \name Constructors
  explicit IndexSpaceIterator(const std::vector<int64_t> &shape,
                              bool lastIndex = false);

  /// Get the current index.
  llvm::ArrayRef<int64_t> operator*() const { return indices_; }

  bool operator==(const IndexSpaceIterator &it) {
    return shape_ == it.shape_ && counter_ == it.counter_;
  }

  bool operator!=(const IndexSpaceIterator &it) { return !(*this == it); }

  /// Increment to the next valid index while iterating over the dimensions in
  /// major-to-minor order.
  IndexSpaceIterator &operator++();
  IndexSpaceIterator operator++(int);

 private:
  std::vector<int64_t> shape_;
  std::vector<int64_t> indices_;
  int64_t counter_;
};

}  // namespace stablehlo
}  // namespace mlir

#endif  // STABLEHLO_REFERENCE_INDEX_H_
