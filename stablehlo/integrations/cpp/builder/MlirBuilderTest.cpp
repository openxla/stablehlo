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

#include "testing/base/public/gunit.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/Register.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/ChloBuilder.h"
#include "stablehlo/integrations/cpp/builder/FuncBuilder.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace mlir {
namespace stablehlo {
namespace {

// Wrap a module builder and register the classes needed
class StablehloModuleBuilder {
 public:
  StablehloModuleBuilder()
      : context_(), module_builder_(context_, UnknownLoc::get(&context_)) {
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

}  // namespace

TEST(MlirBuilderTest, SimpleAdd) {
  std::string expected = R"(module {
  func.func @main(%arg0: tensor<2xi64>) -> tensor<2xi64> {
    %c = stablehlo.constant dense<1> : tensor<i64>
    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i64>) -> tensor<2xi64>
    %1 = stablehlo.add %arg0, %0 : tensor<2xi64>
    return %1 : tensor<2xi64>
  }
})";

  StablehloModuleBuilder mb;
  {  // Build Main Func
    Location funcLoc = fileLineColLoc(mb->getContext(), "main.mlir", 1, 1);
    func::FunctionBuilder fb(mb.get(), "main", funcLoc);
    auto type2xi64 = RankedTensorType::get({2}, fb.getOpBuilder().getI64Type());
    auto arg0 = func::Argument(fb, type2xi64);
    auto cst = stablehlo::Constant(fb, 1);
    auto broadcast = stablehlo::BroadcastInDim(type2xi64, cst, {});
    auto add = stablehlo::Add(arg0, broadcast);
    func::Return(fb, {add});
  }

  OwningOpRef<ModuleOp> module = mb->build();
  EXPECT_EQ(expected, debugString(*module));
}

TEST(MlirBuilderTest, MultipleReturn) {
  std::string expected = R"(module {
  func.func @main(%arg0: tensor<2xi64>) -> (tensor<2xi64>, tensor<2xi64>) {
    %c = stablehlo.constant dense<1> : tensor<2xi64>
    return %arg0, %c : tensor<2xi64>, tensor<2xi64>
  }
})";

  StablehloModuleBuilder mb;
  {  // Build Main Func
    Location funcLoc = fileLineColLoc(mb->getContext(), "main.mlir", 1, 1);
    func::FunctionBuilder fb(mb.get(), "main", funcLoc);
    auto type2xi64 = makeTensorType(fb.getContext(), {2}, ElementType::I64);
    auto arg0 = func::Argument(fb, type2xi64);
    auto cst = stablehlo::Constant(fb, mlir::makeConstant(1L, type2xi64));
    func::Return(fb, {arg0, cst});
  }

  OwningOpRef<ModuleOp> module = mb->build();
  EXPECT_EQ(expected, debugString(*module));
}

TEST(MlirBuilderTest, NoReturn) {
  std::string expected = R"(module {
  func.func @main() {
    return
  }
})";

  StablehloModuleBuilder mb;
  {  // Build Main Func
    Location funcLoc = fileLineColLoc(mb->getContext(), "main.mlir", 1, 1);
    func::FunctionBuilder fb(mb.get(), "main", funcLoc);
    func::Return(fb, {});
  }

  OwningOpRef<ModuleOp> module = mb->build();
  EXPECT_EQ(expected, debugString(*module));
}

TEST(MlirBuilderTest, MixedDialectProgram) {
  std::string expected = R"(module {
  func.func @main(%arg0: tensor<4xi64>) -> tensor<2xi64> {
    %c = stablehlo.constant dense<1> : tensor<i64>
    %0 = chlo.broadcast_add %arg0, %c : (tensor<4xi64>, tensor<i64>) -> tensor<4xi64>
    %values, %indices = chlo.top_k(%0, k = 2) : tensor<4xi64> -> (tensor<2xi64>, tensor<2xi32>)
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i64>) -> tensor<2xi64>
    return %1 : tensor<2xi64>
  }
})";

  StablehloModuleBuilder mb;
  {  // Build Main Func
    func::FunctionBuilder fb(mb.get(), "main");
    auto type4xi64 = makeTensorType(fb.getContext(), {4}, ElementType::I64);
    auto arg0 = func::Argument(fb, type4xi64);
    auto cst = stablehlo::Constant(fb, 1);
    auto add = chlo::BroadcastAdd(arg0, cst);
    auto topkAndIndices = chlo::TopK(add, 2);
    auto broadcast =
        stablehlo::BroadcastInDim(topkAndIndices[0].getType(), cst, {});
    func::Return(fb, broadcast);
  }

  OwningOpRef<ModuleOp> module = mb->build();
  EXPECT_EQ(expected, debugString(*module));
}

TEST(MlirBuilderTest, TestSourceLocation) {
  std::string expected = R"(#loc1 = loc("main.mlir":1:1)
module {
  func.func @main(%arg0: tensor<i64> loc("main.mlir":1:1)) -> tensor<i64> {
    %c = stablehlo.constant dense<1> : tensor<i64> loc(#loc2)
    return %c : tensor<i64> loc(#loc1)
  } loc(#loc1)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("constant.mlir":10:20)
)";

  StablehloModuleBuilder mb;
  {  // Build Main Func
    ScopedBuilderLocation loc(
        mb.get(), fileLineColLoc(mb->getContext(), "main.mlir", 1, 1));
    func::FunctionBuilder fb(mb.get(), "main");
    auto type2xi64 = makeTensorType(fb.getContext(), {}, ElementType::I64);
    auto arg0 = func::Argument(fb, type2xi64);
    static_cast<void>(arg0);  // unused

    // This would typically be a library call, emulate with a lambda.
    auto buildCst = [type2xi64](MlirBuilder& b) {
      ScopedBuilderLocation loc(
          b, fileLineColLoc(b.getContext(), "constant.mlir", 10, 20));
      return stablehlo::Constant(b, mlir::makeConstant(1L, type2xi64));
    };

    func::Return(fb, buildCst(fb));
  }

  OwningOpRef<ModuleOp> module = mb->build();
  std::string moduleString;
  llvm::raw_string_ostream os(moduleString);
  module->print(os, OpPrintingFlags().enableDebugInfo());
  EXPECT_EQ(expected, moduleString);
}

////////
// Region Tests
////////

TEST(MlirBuilderTest, TestOpWithMultipleRegions) {
  std::string expected = R"(module {
  func.func @main(%arg0: tensor<i64>) -> tensor<i64> {
    %c = stablehlo.constant dense<1> : tensor<i64>
    %0 = stablehlo.while(%iterArg = %arg0) : tensor<i64>
    cond {
      %1 = stablehlo.compare  LT, %iterArg, %c : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
    } do {
      %1 = stablehlo.subtract %iterArg, %c : tensor<i64>
      stablehlo.return %1 : tensor<i64>
    }
    return %0 : tensor<i64>
  }
})";

  StablehloModuleBuilder mb;
  func::FunctionBuilder fb(mb.get(), "main");
  auto arg0Type = makeTensorType(fb.getContext(), {}, ElementType::I64);
  auto arg0 = func::Argument(fb, arg0Type);

  auto cst = stablehlo::Constant(fb, mlir::makeConstant(1L, arg0Type));

  auto loop = stablehlo::While(
      fb, {arg0},
      [&cst](RegionBuilder& cond) {
        auto loopArg0 = Argument(cond, cst.getType());
        auto cmp = stablehlo::Compare(loopArg0, cst,
                                      stablehlo::ComparisonDirection::LT);
        stablehlo::Return(cond, cmp);
      },
      [&cst](RegionBuilder& body) {
        auto loopArg0 = Argument(body, cst.getType());
        auto add = stablehlo::Subtract(loopArg0, cst);
        stablehlo::Return(body, add);
      });
  func::Return(fb, loop);

  OwningOpRef<ModuleOp> module = mb->build();
  EXPECT_TRUE(succeeded(mlir::verify(module.get())));
  EXPECT_EQ(expected, debugString(*module));
}

////////
// Func Dialect Tests
////////

TEST(FuncBuilderTest, TestFuncCallbackApi) {
  std::string expected = R"(module {
  func.func @main(%arg0: tensor<i64>, %arg1: tensor<i64>) -> tensor<i64> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<i64>
    return %0 : tensor<i64>
  }
})";

  StablehloModuleBuilder mb;
  func::Func(mb.get(), "main", [](RegionBuilder& rb) {
    auto type = makeTensorType(rb.getContext(), {}, ElementType::I64);
    auto regArg0 = Argument(rb, type);
    auto regArg1 = Argument(rb, type);
    auto add = Add(regArg0, regArg1);
    func::Return(rb, add);
  });

  OwningOpRef<ModuleOp> module = mb->build();
  EXPECT_EQ(expected, debugString(*module));
}

TEST(FuncBuilderTest, TestCallOp) {
  std::string expected = R"(module {
  func.func @callee(%arg0: tensor<i64>, %arg1: tensor<i64>) -> tensor<i64> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<i64>
    return %0 : tensor<i64>
  }
  func.func @main(%arg0: tensor<i64>, %arg1: tensor<i64>) -> tensor<i64> {
    %0 = call @callee(%arg0, %arg1) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    return %0 : tensor<i64>
  }
})";

  StablehloModuleBuilder mb;

  // Build subfunction
  func::FuncOp callee;
  auto type = makeTensorType(mb->getContext(), {}, ElementType::I64);
  {
    func::FunctionBuilder fb(mb.get(), "callee");
    auto regArg0 = func::Argument(fb, type);
    auto regArg1 = func::Argument(fb, type);
    auto add = Add(regArg0, regArg1);
    func::Return(fb, add);
    callee = fb.build();
  }

  // Build main function
  {
    func::FunctionBuilder fb(mb.get(), "main");
    auto arg0 = func::Argument(fb, type);
    auto arg1 = func::Argument(fb, type);
    auto call = func::Call(fb, callee, {arg0, arg1});
    func::Return(fb, call);
  }

  OwningOpRef<ModuleOp> module = mb->build();
  EXPECT_EQ(expected, debugString(*module));
}

}  // namespace stablehlo
}  // namespace mlir
