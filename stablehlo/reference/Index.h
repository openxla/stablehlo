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
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace stablehlo {

/// Iterates over the index space of a tensor with a given shape, producing
/// indices in lexicographical order. As an example, for a tensor with shape
/// [2,3], the iterator enumerates the indices (0,0), (0,1), (0,2), (1,0),
/// (1,1), (1,2) and <END> (special past-the-end element which cannot be
/// dereferenced).
class IndexSpaceIterator {
 public:
  /// \name Constructor
  IndexSpaceIterator(llvm::ArrayRef<int64_t> shape,
                     llvm::Optional<llvm::SmallVector<int64_t>> index)
      : shape_(shape), index_(index){};

  /// Get the current index.
  /// At any point in time, the iterator can either reference an actual index
  /// or the past-the-end element in the index space.
  /// Dereferencing a past-the-end iterator will result in a fatal error.
  llvm::ArrayRef<int64_t> operator*() const;

  /// Compare the iterator to another iterator.
  /// Two iterators are equal if they have the same underlying shape and
  /// reference the same element in the index space.
  bool operator==(const IndexSpaceIterator &it) {
    return shape_ == it.shape_ && index_ == it.index_;
  }
  bool operator!=(const IndexSpaceIterator &it) { return !(*this == it); }

  /// Increment to the next index while iterating over the index space
  /// of a tensor in lexicographical order.
  /// Incrementing past the last index will result in a past-the-end iterator
  /// which cannot be dereferenced. Incrementing even further will result in
  /// a fatal error.
  /// For scalar tensor, which empty 'shape_', incrementing the index result in
  /// past-the-end iterator.
  IndexSpaceIterator &operator++();
  IndexSpaceIterator operator++(int);

 private:
  /// Shape of the tensor whose index space to be iterated on.
  llvm::SmallVector<int64_t> shape_;

  /// Current multi-dimensional index.
  /// If the optional is empty, then we're at the end
  llvm::Optional<llvm::SmallVector<int64_t>> index_;
};

}  // namespace stablehlo
}  // namespace mlir

#endif  // STABLEHLO_REFERENCE_INDEX_H_
