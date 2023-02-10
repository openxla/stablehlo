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

#ifndef STABLEHLO_REFERENCE_SCOPE_H_
#define STABLEHLO_REFERENCE_SCOPE_H_

#include "stablehlo/reference/Tensor.h"

namespace mlir {
namespace stablehlo {

/// Represents the runtime scope corresponding to a region of a program under
/// evaluation. Holds (1) mapping from SSA values, defined in the current
/// region, to their evaluated runtime `Tensor` values, and (2) handle to
/// `InterpreterScope` object corresponding to the syntactically enclosing
/// region.
///
/// Note:
/// 1. A `InterpreterScope` object is instantiated every time a region is
///    evaluated.
/// 2. A `InterpreterScope` object treats the `parentScope` as immutable to
///    align with the fact that a StableHLO program, in pure SSA form (without
///    memory allocation/load/store ops), disallows mutating the `parentScope`
///    object from within a region.
class InterpreterScope {
 public:
  InterpreterScope(const InterpreterScope *const pScope)
      : parentScope(pScope) {}

  InterpreterScope(const InterpreterScope &Scope) = delete;

  /// Add the mapping from SSA value (`ssaValue`), defined in a region, to
  /// its evaluated runtime value (`runtimeValue`).
  void add(Value ssaValue, Tensor runtimeValue);

  /// Find the runtime value mapped to SSA value `ssaValue`. The search starts
  /// with the current scope and then recursively continues over to the scope
  /// defined by `parentScope`.
  Tensor find(Value ssaValue) const;

 private:
  /// Internal store for mapping from SSA values to runtime `Tensor` values.
  llvm::DenseMap<Value, Tensor> stackFrame;

  /// A handle to the parent's scope.
  const InterpreterScope *const parentScope = nullptr;
};

}  // namespace stablehlo
}  // namespace mlir

#endif  // STABLEHLO_REFERENCE_SCOPE_H_
