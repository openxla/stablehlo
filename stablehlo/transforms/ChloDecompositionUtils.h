/* Copyright 2024 The StableHLO Authors. All Rights Reserved.
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

#ifndef STABLEHLO_TRANSFORMS_CHLO_DECOMP_UTILS_H_
#define STABLEHLO_TRANSFORMS_CHLO_DECOMP_UTILS_H_

#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace stablehlo {

// Utility functions used in the Chlo to stablehlo legalization.

Value materializeLgamma(OpBuilder &rewriter, Location loc, ValueRange args);

Value materializeDigamma(OpBuilder &rewriter, Location loc, ValueRange args);

Value materializeZeta(OpBuilder &rewriter, Location loc, ValueRange args);

Value materializePolygamma(OpBuilder &rewriter, Location loc, ValueRange args);

}  // namespace stablehlo
}  // namespace mlir

#endif  // STABLEHLO_TRANSFORMS_CHLO_DECOMP_UTILS_H_
