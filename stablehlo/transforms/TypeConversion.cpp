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

#include "stablehlo/transforms/TypeConversion.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"

namespace mlir {
namespace vhlo {

void registerFuncOpsForTypeConversion(ConversionTarget& target,
                                      RewritePatternSet& patterns,
                                      TypeConverter& converter) {
  target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
    return converter.isSignatureLegal(op.getFunctionType());
  });
  target.addDynamicallyLegalOp<func::CallOp>([&](func::CallOp op) {
    return converter.isSignatureLegal(op.getCalleeType());
  });
  target.addDynamicallyLegalOp<func::ReturnOp>([&](func::ReturnOp op) {
    return converter.isLegal(op.getOperandTypes());
  });
  populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns,
                                                                 converter);
  populateCallOpTypeConversionPattern(patterns, converter);
  populateReturnOpTypeConversionPattern(patterns, converter);
}

}  // namespace vhlo
}  // namespace mlir
