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

#ifndef STABLEHLO_REFERENCE_SCOPE_H_
#define STABLEHLO_REFERENCE_SCOPE_H_

#include "stablehlo/reference/Tensor.h"

namespace mlir {
namespace stablehlo {

/// Represents the scope corresponding to a region of a program under
/// evaluation. Holds (1) mapping from SSA values, defined in the current
/// region, to their evaluated runtime `Tensor` values, and (2) handle to
/// `Scope` object corresponding to the syntactically enclosing region.
class Scope {
 public:
  Scope(Scope *parent) : parent_(parent) {}

  Scope(Scope &&other) = default;
  Scope &operator=(Scope &&other) = default;

  Scope(const Scope &) = delete;
  Scope &operator=(const Scope &) = delete;

  /// Add the mapping from SSA value (`ssaValue`), defined in a region, to
  /// its evaluated runtime value (`runtimeValue`).
  void add(Value ssaValue, Tensor runtimeValue);

  /// Add the mapping from SSA values (`ssaValues`), defined in a region, to
  /// its evaluated runtime values (`runtimeValues`).
  void add(ValueRange ssaValues, ArrayRef<Tensor> runtimeValues);

  /// Find the runtime value mapped to SSA value `ssaValue`. The search starts
  /// with the current scope and then recursively continues over to the scope
  /// defined by `parent_`.
  Tensor find(Value ssaValue) const;

  /// Find the runtime values mapped to SSA values `ssaValues`.
  // SmallVector<Tensor> find(ArrayRef<Value> ssaValues) const;
  SmallVector<Tensor> find(ValueRange ssaValues) const;

 private:
  /// Internal store for mapping from SSA values to runtime `Tensor` values.
  llvm::DenseMap<Value, Tensor> stack_frame_;

  /// A handle to the parent's scope.
  Scope *parent_;
};

}  // namespace stablehlo
}  // namespace mlir

#endif  // STABLEHLO_REFERENCE_SCOPE_H_
