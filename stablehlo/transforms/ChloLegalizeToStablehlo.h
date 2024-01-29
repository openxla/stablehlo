/* Copyright 2020 The OpenXLA Authors.

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

#ifndef STABLEHLO_TRANSFORMS_CHLO_LEGALIZE_TO_STABLEHLO_H
#define STABLEHLO_TRANSFORMS_CHLO_LEGALIZE_TO_STABLEHLO_H

#include "mlir/IR/PatternMatch.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {
namespace stablehlo {

template <typename FromOpTy, typename ToOpTy>
struct HloNaryElementwiseAdaptor {
  static ToOpTy createOp(FromOpTy fromOp, Type resultType,
                         ValueRange broadcastedOperands, OpBuilder &builder) {
    return builder.create<ToOpTy>(fromOp.getLoc(), resultType,
                                  broadcastedOperands);
  }
};

template <template <typename, typename, typename> class Pattern,
          typename... ConstructorArgs>
void populateForBroadcastingBinaryOp(MLIRContext *context,
                                     RewritePatternSet *patterns,
                                     ConstructorArgs &&...args) {
#define POPULATE_BCAST(ChloOp, StablehloOp)                               \
  patterns->add<Pattern<ChloOp, StablehloOp,                              \
                        HloNaryElementwiseAdaptor<ChloOp, StablehloOp>>>( \
      context, args...);

  POPULATE_BCAST(chlo::BroadcastAddOp, stablehlo::AddOp);

#undef POPULATE_BCAST
}
}  // namespace stablehlo
}  // namespace mlir

#endif  // STABLEHLO_TRANSFORMS_CHLO_LEGALIZE_TO_STABLEHLO_H