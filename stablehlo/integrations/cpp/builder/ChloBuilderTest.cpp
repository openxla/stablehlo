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

#include <string>

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/Register.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/ChloBuilder.h"
#include "stablehlo/integrations/cpp/builder/FuncBuilder.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"
#include "testing/base/public/gunit.h"

namespace mlir {
namespace chlo {

namespace {

// Wrap a module builder and register the classes needed
class ChloModuleBuilder {
 public:
  ChloModuleBuilder()
      : context_(), module_builder_(context_, mlir::unknownLoc(context_)) {
    DialectRegistry registry;
    stablehlo::registerAllDialects(registry);
    context_.appendDialectRegistry(registry);
    context_.loadAllAvailableDialects();
  }

  ModuleBuilder& get() { return module_builder_; }
  ModuleBuilder* operator->() { return &module_builder_; }

 private:
  MLIRContext context_;
  ModuleBuilder module_builder_;
};

// TODO: Make a FileCheck matcher

}  // namespace

TEST(ChloBuilderTest, SmokeTest) {
  std::string expected = R"mlir(module {
  func.func @main(%arg0: tensor<2xi64>) -> tensor<2xi64> {
    %0 = chlo.constant dense<1> : tensor<i64>
    %1 = chlo.broadcast_add %arg0, %0 : (tensor<2xi64>, tensor<i64>) -> tensor<2xi64>
    return %1 : tensor<2xi64>
  }
})mlir";

  ChloModuleBuilder mb;
  {  // Build Main Func
    Location funcLoc = fileLineColLoc(mb->getContext(), "main.mlir", 1, 1);
    func::FunctionBuilder fb(mb.get(), "main", funcLoc);
    auto type2xi64 = makeTensorType(mb->getContext(), {2}, ElementType::I64);
    auto typeScalari64 = makeTensorType(mb->getContext(), {}, ElementType::I64);
    auto arg0 = func::Argument(fb, type2xi64);
    auto cst = Constant(fb, mlir::makeConstant(1L, typeScalari64));
    auto add = BroadcastAdd(arg0, cst);
    func::Return(fb, {add});
  }

  OwningOpRef<ModuleOp> module = mb->build();
  EXPECT_TRUE(succeeded(mlir::verify(*module)));
  EXPECT_EQ(expected, debugString(*module));
}

TEST(MlirBuilderTest, ConstantLike) {
  std::string expected = R"mlir(module {
  func.func @main(%arg0: tensor<2xi64>) -> tensor<2xi64> {
    %0 = "chlo.constant_like"(%arg0) <{value = 1 : i64}> : (tensor<2xi64>) -> tensor<2xi64>
    return %0 : tensor<2xi64>
  }
})mlir";

  ChloModuleBuilder mb;
  {  // Build Main Func
    Location funcLoc = fileLineColLoc(mb->getContext(), "main.mlir", 1, 1);
    func::FunctionBuilder fb(mb.get(), "main", funcLoc);
    auto type2xi64 = makeTensorType(mb->getContext(), {2}, ElementType::I64);
    auto typeScalari64 = makeTensorType(mb->getContext(), {}, ElementType::I64);
    auto arg0 = func::Argument(fb, type2xi64);
    auto cst = ConstantLike(arg0, mlir::makeConstant(1L, typeScalari64));
    func::Return(fb, {cst});
  }

  OwningOpRef<ModuleOp> module = mb->build();
  EXPECT_TRUE(succeeded(mlir::verify(*module)));
  EXPECT_EQ(expected, debugString(*module));
}

TEST(MlirBuilderTest, ConstantLikeBounded) {
  std::string expected = R"mlir(module {
  func.func @main(%arg0: tensor<2xi64>, %arg1: tensor<i32>) -> tensor<?xi32, #stablehlo.bounds<2>> {
    %0 = stablehlo.set_dimension_size %arg0, %arg1, dim = 0 : (tensor<2xi64>, tensor<i32>) -> tensor<?xi64, #stablehlo.bounds<2>>
    %1 = "chlo.constant_like"(%0) <{value = 1 : i32}> : (tensor<?xi64, #stablehlo.bounds<2>>) -> tensor<?xi32, #stablehlo.bounds<2>>
    return %1 : tensor<?xi32, #stablehlo.bounds<2>>
  }
})mlir";

  ChloModuleBuilder mb;
  {  // Build Main Func
    Location funcLoc = fileLineColLoc(mb->getContext(), "main.mlir", 1, 1);
    func::FunctionBuilder fb(mb.get(), "main", funcLoc);
    auto type2xi64 = makeTensorType(mb->getContext(), {2}, ElementType::I64);
    auto typei32 = makeTensorType(mb->getContext(), {}, ElementType::I32);
    auto arg0 = func::Argument(fb, type2xi64);
    auto arg1 = func::Argument(fb, typei32);
    auto sds = stablehlo::SetDimensionSize(arg0, arg1, 0);
    auto cst = ConstantLike(sds, mlir::makeConstant(1L, typei32));
    func::Return(fb, {cst});
  }

  OwningOpRef<ModuleOp> module = mb->build();
  EXPECT_TRUE(succeeded(mlir::verify(*module)));
  EXPECT_EQ(expected, debugString(*module));
}

}  // namespace chlo
}  // namespace mlir
