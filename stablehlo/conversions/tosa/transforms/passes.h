/* Copyright 2022 OpenXLA Authors. All Rights Reserved.

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

#ifndef STABLEHLO_CONVERSIONS_TOSA_TRANSFORMS_PASSES_H
#define STABLEHLO_CONVERSIONS_TOSA_TRANSFORMS_PASSES_H

#include <memory>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace tosa {

std::unique_ptr<OperationPass<func::FuncOp>> createLegalizeStablehloPass();
std::unique_ptr<OperationPass<func::FuncOp>> createPrepareStablehloPass();

#define GEN_PASS_REGISTRATION
#define GEN_PASS_DECL_TOSALEGALIZESTABLEHLOPASS
#include "stablehlo/conversions/tosa/transforms/passes.h.inc"

}  // namespace tosa
}  // namespace mlir

#endif  // STABLEHLO_CONVERSIONS_TOSA_TRANSFORMS_PASSES_H
