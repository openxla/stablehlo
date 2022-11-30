/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.
   Copyright 2022 The StableHLO Authors.

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

#ifndef STABLEHLO_COMPATIBILITY_DIALECT_REGISTER_H
#define STABLEHLO_COMPATIBILITY_DIALECT_REGISTER_H

#include "mlir/IR/DialectRegistry.h"

namespace mlir {
namespace vhlo {

// Add vhlo dialects to the provided registry.
void registerAllDialects(DialectRegistry &registry);

}  // namespace vhlo
}  // namespace mlir

#endif  // STABLEHLO_COMPATIBILITY_DIALECT_REGISTER_H
