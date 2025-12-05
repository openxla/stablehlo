/* Copyright 2025 The OpenXLA Authors.

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

#ifndef STABLEHLO_BUILDER_CHLOBUILDER_H_
#define STABLEHLO_BUILDER_CHLOBUILDER_H_

#include <cstdint>

#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"

namespace mlir {
namespace chlo {

/////////////////
// GENERATED APIs
/////////////////

#include "stablehlo/integrations/cpp/builder/ChloBuilder.h.inc"

/////////////////
// MANUAL APIs
/////////////////

MlirOp ConstantLike(MlirOp input, DenseElementsAttr val);

}  // namespace chlo
}  // namespace mlir

#endif  // STABLEHLO_BUILDER_CHLOBUILDER_H_
