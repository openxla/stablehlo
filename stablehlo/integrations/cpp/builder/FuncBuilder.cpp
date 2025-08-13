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

#include "stablehlo/integrations/cpp/builder/FuncBuilder.h"

#include <cstdint>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"

namespace mlir {
namespace func {

void updateFuncSignature(FuncOp func) {
  TypeRange argTypes = func.getBody().getArgumentTypes();
  TypeRange retTypes{};

  auto ret = func.getOps<ReturnOp>();
  if (!ret.empty()) {
    ReturnOp retOp = *ret.begin();
    retTypes = retOp.getOperandTypes();
  }

  func.setFunctionType(
      FunctionType::get(func.getContext(), argTypes, retTypes));
}

// Must be called after arguments or return types are changed.
void FunctionBuilder::notifySignatureChanged() {
  FuncOp func = getOp();
  updateFuncSignature(func);
}

MlirOp Argument(FunctionBuilder& fb, Type type) {
  RegionBuilder rb = fb.getRegionBuilder();
  MlirOp arg = ::mlir::Argument(rb, type);
  fb.notifySignatureChanged();

  // RegionBuilder will go out of scope, so swap the arg builder to the
  // FunctionBuilder.
  return swap(fb, arg);
}

void Return(FunctionBuilder& fb, MlirOp& value) {
  return Return(fb, ArrayRef<MlirOp>{value});
}

void Return(FunctionBuilder& fb, ArrayRef<MlirOp> values) {
  RegionBuilder rb = fb.getRegionBuilder();
  Return(rb, values);
  fb.notifySignatureChanged();
}

SmallVector<MlirOp> Call(MlirBuilder& builder, func::FuncOp func,
                         ArrayRef<MlirOp> operands) {
  return builder.createVariadic<CallOp>(func, unwrap(operands));
}

/////////////////
// GENERATED APIs
/////////////////

// Idea: Update function signature in func::Return.
// May need to specialize the op gen for func return...

FuncOp Func(MlirBuilder& mb, StringRef name,
            const RegionBuilderCallback& body) {
  FuncOp func = mb.createUnwrapped<func::FuncOp>(
      name, FunctionType::get(&mb.getContext(), {}, {}));
  RegionBuilder rb(mb, func->getRegion(0));
  body(rb);
  updateFuncSignature(func);
  return func;
}

#include "stablehlo/integrations/cpp/builder/FuncBuilder.cpp.inc"

}  // namespace func
}  // namespace mlir
