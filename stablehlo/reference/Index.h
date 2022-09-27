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
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace stablehlo {

/// Iterates over the index space of a tensor with a given shape, producing
/// indices in lexicographical order. As an example, for a tensor with shape
/// [2,3], the iterator enumerates the indices (0,0), (0,1), (0,2), (1,0),
/// (1,1), and (1,2).
class IndexSpaceIterator {
 public:
  /// \name Constructor
  explicit IndexSpaceIterator(llvm::ArrayRef<int64_t> shape, int64_t counter);

  /// Get the current indices.
  llvm::ArrayRef<int64_t> operator*() const { return indices_; }

  bool operator==(const IndexSpaceIterator &it) {
    return shape_ == it.shape_ && counter_ == it.counter_;
  }

  bool operator!=(const IndexSpaceIterator &it) { return !(*this == it); }

  /// Increment to the next valid index while iterating over the index space
  /// of a tensor in lexicographical order. Incrementing beyond the last index
  /// results in fatal error.
  IndexSpaceIterator &operator++();
  IndexSpaceIterator operator++(int);

 private:
  /// Shape of the tensor whose index space to be iterated on.
  llvm::SmallVector<int64_t> shape_;

  /// Current multi-dimentional index.
  llvm::SmallVector<int64_t> indices_;

  /// Counter for the number of indices iterated. This is also used to set-up
  /// past-the-end iterator.
  int64_t counter_;

  /// For a tensor with shape 'shape_', 'num_elements_' provodes the total
  /// number of indices to iterate on. Helps to flag increments beyong the last
  /// (in lexicographical order) indices.
  int64_t num_elements_;
};

}  // namespace stablehlo
}  // namespace mlir

#endif  // STABLEHLO_REFERENCE_INDEX_H_
