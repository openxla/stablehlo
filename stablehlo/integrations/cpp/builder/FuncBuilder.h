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

#ifndef STABLEHLO_BUILDER_FUNCBUILDER_H_
#define STABLEHLO_BUILDER_FUNCBUILDER_H_

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"

namespace mlir {
namespace func {

class FunctionBuilder;

// Functional wrappers for use in builder methods.
MlirOp Argument(FunctionBuilder& fb, Type type);
void Return(FunctionBuilder& fb, MlirOp& value);
void Return(FunctionBuilder& fb, ArrayRef<MlirOp> values);

// TODO: Do we need RegionOpBuilder? Region ops may be too custom to benefit
// from a generic base class.
class FunctionBuilder : public RegionOpBuilder<func::FuncOp> {
 public:
  FunctionBuilder(ModuleBuilder& mb, std::string name,
                  std::optional<Location> loc = std::nullopt)
      : RegionOpBuilder(mb,
                        func::FuncOp::create(
                            mb.getOpBuilder(), loc.value_or(mb.getLoc()), name,
                            FunctionType::get(&mb.getContext(), {}, {}))) {}

  StringRef getName() { return getOp().getName(); }

  void notifySignatureChanged() override;
};

// FuncOp builder, return the raw FuncOp. For a main function this value can
// be ignored, but return the FuncOp since the handle can be stored for later
// use with `Call` to infer output types based on function signature.
func::FuncOp Func(MlirBuilder& mb, StringRef name,
                  const RegionBuilderCallback& body);

SmallVector<MlirOp> Call(MlirBuilder& builder, func::FuncOp func,
                         ArrayRef<MlirOp> operands);

/////////////////
// GENERATED APIs
/////////////////

// AVOID THE GENERATED FUNC API.
// It has worse UX and needs some fixes for func::Return to work with it.
// Currently func::Return doesn't update the function signature by default.

#include "stablehlo/integrations/cpp/builder/FuncBuilder.h.inc"

// GENERATED CODE


}  // namespace func

}  // namespace mlir

#endif  // STABLEHLO_BUILDER_FUNCBUILDER_H_
