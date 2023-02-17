/* Copyright 2022 The StableHLO Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permutationissions and
limitations under the License.
==============================================================================*/

#include "stablehlo/reference/Index.h"

#include "llvm/Support/Error.h"

namespace mlir {
namespace stablehlo {

Index Index::operator+(const Index &other) const {
  if (size() != other.size())
    llvm::report_fatal_error("Index add expects operands of same size.");

  SmallVector<int64_t> result;
  for (auto [lhsIdx, rhsIdx] : llvm::zip(index_, other.index_))
    result.push_back(lhsIdx + rhsIdx);
  return Index(result);
}

Index Index::operator+(ArrayRef<int64_t> array) const {
  if (size() != array.size())
    llvm::report_fatal_error("Index add expects operands of same size.");

  SmallVector<int64_t> result;
  for (auto [lhsIdx, rhsIdx] : llvm::zip(index_, array))
    result.push_back(lhsIdx + rhsIdx);
  return Index(result);
}

Index Index::operator*(const Index &other) const {
  if (size() != other.size())
    llvm::report_fatal_error("Index product expects operands of same size.");

  SmallVector<int64_t> result;
  for (auto [lhsIdx, rhsIdx] : llvm::zip(index_, other.index_))
    result.push_back(lhsIdx * rhsIdx);
  return Index(result);
}

Index Index::operator*(ArrayRef<int64_t> array) const {
  if (size() != array.size())
    llvm::report_fatal_error("Index product expects operands of same size.");

  SmallVector<int64_t> result;
  for (auto [lhsIdx, rhsIdx] : llvm::zip(index_, array))
    result.push_back(lhsIdx * rhsIdx);
  return Index(result);
}

LogicalResult verifyIndex(llvm::ArrayRef<int64_t> shape, const Index &index) {
  if (shape.size() != index.size()) return failure();

  for (auto [shapeDim, indexDim] : llvm::zip(shape, index.getIndexArray()))
    if (indexDim < 0 || indexDim >= shapeDim) return failure();

  return success();
}

Index IndexSpaceIterator::operator*() const {
  if (!index_)
    llvm::report_fatal_error("Dereferencing a past-the-end iterator.");
  return *index_;
}

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

Index Index::permute(ArrayRef<int64_t> permutation) {
  if (size() != permutation.size())
    llvm::report_fatal_error(
        "Index permutationute expects permutation of same size as the index.");

  Index result(size());
  for (size_t i = 0; i < permutation.size(); i++)
    result[i] = (*this)[permutation[i]];
  return result;
}

Index operator+(ArrayRef<int64_t> array, const Index &index) {
  return index + array;
}

Index operator*(ArrayRef<int64_t> array, const Index &index) {
  return index * array;
}

}  // namespace stablehlo
}  // namespace mlir
