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

class Scope {
 public:
  Scope(const Scope *const pScope) : parentScope(pScope) {}

  Scope(const Scope &Scope) = delete;

  void add(Value ssaValue, Tensor runtimeValue);
  Tensor fetchOperandInScope(Value value) const;

 private:
  llvm::DenseMap<Value, Tensor> stackFrame;
  const Scope *parentScope;
};

}  // namespace stablehlo
}  // namespace mlir

#endif  // STABLEHLO_REFERENCE_SCOPE_H_
