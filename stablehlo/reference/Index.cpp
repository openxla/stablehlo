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

#include "stablehlo/reference/Index.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace stablehlo {

raw_ostream &operator<<(raw_ostream &os, const Sizes &x) {
  os << "[";
  llvm::interleave(x, os, ", ");
  os << "]";
  return os;
}

Sizes Sizes::permute(ArrayRef<int64_t> permutation) const {
  Sizes result(size());
  for (size_t i = 0; i < permutation.size(); i++)
    result[i] = (*this)[permutation[i]];
  return result;
}

bool Sizes::inBounds(const Sizes &bounds) const {
  if (size() != bounds.size()) return false;
  for (auto [size, bound] : llvm::zip(*this, bounds))
    if (size < 0 || size >= bound) return false;
  return true;
}

IndexSpaceIterator Sizes::index_begin() const {
  if (any_of(*this, [](int64_t dimSize) { return dimSize == 0; }))
    return IndexSpaceIterator(*this);

  Index initialIndex(size());
  return IndexSpaceIterator(*this, initialIndex);
}

IndexSpaceIterator Sizes::index_end() const {
  return IndexSpaceIterator(*this);
}

Sizes operator+(const Sizes &x, const Sizes &y) {
  if (x.size() != y.size()) llvm::report_fatal_error("expected same size");
  Sizes result(x.size());
  for (size_t i = 0; i < x.size(); ++i) {
    result[i] = x[i] + y[i];
  }
  return result;
}

Sizes operator+(const Sizes &x, int64_t y) { return x + Sizes(x.size(), y); }

Sizes operator+(int64_t x, const Sizes &y) { return y + x; }

Sizes operator-(const Sizes &x, const Sizes &y) {
  if (x.size() != y.size()) llvm::report_fatal_error("expected same size");
  Sizes result(x.size());
  for (size_t i = 0; i < x.size(); ++i) {
    result[i] = x[i] - y[i];
  }
  return result;
}

Sizes operator-(const Sizes &x, int64_t y) { return x - Sizes(x.size(), y); }

Sizes operator-(int64_t x, const Sizes &y) { return Sizes(y.size(), x) - y; }

Sizes operator*(const Sizes &x, const Sizes &y) {
  if (x.size() != y.size()) llvm::report_fatal_error("expected same size");
  Sizes result(x.size());
  for (size_t i = 0; i < x.size(); ++i) {
    result[i] = x[i] * y[i];
  }
  return result;
}

Sizes operator*(const Sizes &x, int64_t y) { return x * Sizes(x.size(), y); }

Sizes operator*(int64_t &x, const Sizes &y) { return y + x; }

Sizes clamp(int64_t min, const Sizes &x, int64_t max) {
  return clamp(Sizes(x.size(), min), x, Sizes(x.size(), max));
}

Sizes clamp(int64_t min, const Sizes &x, const Sizes &max) {
  return clamp(Sizes(x.size(), min), x, max);
}

Sizes clamp(const Sizes &min, const Sizes &x, int64_t max) {
  return clamp(min, x, Sizes(x.size(), max));
}

Sizes clamp(const Sizes &min, const Sizes &x, const Sizes &max) {
  if (min.size() != x.size() || x.size() != max.size())
    llvm::report_fatal_error("expected same size");
  Sizes result(x.size());
  for (size_t i = 0; i < x.size(); ++i) {
    result[i] = std::min(std::max(x[i], min[i]), max[i]);
  }
  return result;
}

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
