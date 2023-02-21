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

#ifndef STABLEHLO_REFERENCE_SIZES_H
#define STABLEHLO_REFERENCE_SIZES_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace mlir {
namespace stablehlo {

/// Represents per axis metadata (e.g. tensor shape, slice sizes etc.) of type
/// `int64_t`.
class Sizes : public SmallVector<int64_t> {
 public:
  Sizes() = default;
  Sizes(const Sizes &other) = default;
  Sizes &operator=(const Sizes &other) = default;

  Sizes(std::initializer_list<int64_t> list) : SmallVector(list) {}
  explicit Sizes(size_t size, int64_t element = 0)
      : SmallVector(size, element) {}
  explicit Sizes(ArrayRef<int64_t> array) : SmallVector(array) {}
  explicit Sizes(DenseIntElementsAttr attr)
      : SmallVector(attr.getValues<int64_t>()) {}

  // Returns `s` with the effect of applying `permutation`
  // to `this` object, that is, `s[i] = (*this)[permutation[i]]`.
  Sizes permute(ArrayRef<int64_t> permutation) const;

  /// Checks if an element `e` at kth axis of `this` object follows
  /// `0 <= e <= bounds[k]`.
  bool inBounds(const Sizes &bounds) const;
};

raw_ostream &operator<<(raw_ostream &os, const Sizes &x);

/// Overloaded add operator to return `Sizes` object `z` such that
/// `z[k] = x[k] + y[k]` for all axis k.
Sizes operator+(const Sizes &x, const Sizes &y);

/// Overloaded add operator to return `Sizes` object `z` such that
/// `z[k] = x[k] + y` for all axis k.
Sizes operator+(const Sizes &x, int64_t y);

/// Overloaded add operator to return `Sizes` object `z` such that
/// `z[k] = x + y[k]` for all axis k.
Sizes operator+(int64_t x, const Sizes &y);

/// Overloaded add operator to return `Sizes` object `z` such that
/// `z[k] = x[k] - y[k]` for all axis k.
Sizes operator-(const Sizes &x, const Sizes &y);

/// Overloaded add operator to return `Sizes` object `z` such that
/// `z[k] = x[k] - y` for all axis k.
Sizes operator-(const Sizes &x, int64_t y);

/// Overloaded add operator to return `Sizes` object `z` such that
/// `z[k] = x - y[k]` for all axis k.
Sizes operator-(int64_t x, const Sizes &y);

/// Overloaded add operator to return `Sizes` object `z` such that
/// `z[k] = x[k] * y[k]` for all axis k.
Sizes operator*(const Sizes &x, const Sizes &y);

/// Overloaded add operator to return `Sizes` object `z` such that
/// `z[k] = x[k] * y` for all axis k.
Sizes operator*(const Sizes &x, int64_t y);

/// Overloaded add operator to return `Sizes` object `z` such that
/// `z[k] = x * y[k]` for all axis k.
Sizes operator*(int64_t x, const Sizes &y);

/// Clamp operator to return `Sizes` object `z` such that
/// `z[k] = std::min(std::max(x[k], min[k]), max[k])` for all axis k.
Sizes clamp(const Sizes &min, const Sizes &x, const Sizes &max);

/// Clamp operator to return `Sizes` object `z` such that
/// `z[k] = std::min(std::max(x[k], min), max)` for all axis k.
Sizes clamp(int64_t min, const Sizes &x, int64_t max);

/// Clamp operator to return `Sizes` object `z` such that
/// `z[k] = std::min(std::max(x[k], min), max[k])` for all axis k.
Sizes clamp(int64_t min, const Sizes &x, const Sizes &max);

/// Clamp operator to return `Sizes` object `z` such that
/// `z[k] = std::min(std::max(x[k], min[k]), max)` for all axis k.
Sizes clamp(const Sizes &min, const Sizes &x, int64_t max);

/// Represents index of a tensor.
using Index = Sizes;

}  // namespace stablehlo
}  // namespace mlir

#endif  // STABLEHLO_REFERENCE_SIZES_H
