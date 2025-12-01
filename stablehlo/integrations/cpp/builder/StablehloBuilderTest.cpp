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

#include <complex>
#include <cstdint>
#include <string>

#include "gtest/gtest.h"
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
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
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
      : context_(), module_builder_(context_, mlir::unknownLoc(context_)) {
    DialectRegistry registry;
    registerAllDialects(registry);
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

TEST(MlirBuilderTest, SmokeTest) {
  std::string expected = R"mlir(module {
  func.func @main(%arg0: tensor<2xi64>) -> tensor<2xi64> {
    %c = stablehlo.constant dense<1> : tensor<2xi64>
    %0 = stablehlo.add %arg0, %c : tensor<2xi64>
    return %0 : tensor<2xi64>
  }
})mlir";

  StablehloModuleBuilder mb;
  {  // Build Main Func
    Location funcLoc = fileLineColLoc(mb->getContext(), "main.mlir", 1, 1);
    func::FunctionBuilder fb(mb.get(), "main", funcLoc);
    auto type2xi64 = makeTensorType(mb->getContext(), {2}, ElementType::I64);
    auto arg0 = func::Argument(fb, type2xi64);
    auto cst = Constant(fb, mlir::makeConstant(1L, type2xi64));
    auto add = Add(arg0, cst);
    func::Return(fb, {add});
  }

  // TODO: Make these a FileCheck based test.
  OwningOpRef<ModuleOp> module = mb->build();
  EXPECT_TRUE(succeeded(mlir::verify(*module)));
  EXPECT_EQ(expected, debugString(*module));
}

TEST(MlirBuilderTest, BinaryOps) {
  std::string expected = R"mlir(module {
  func.func @main(%arg0: tensor<2xi64>) -> tensor<2xi64> {
    %c = stablehlo.constant dense<1> : tensor<2xi64>
    %0 = stablehlo.add %arg0, %c : tensor<2xi64>
    %1 = stablehlo.subtract %arg0, %0 : tensor<2xi64>
    %2 = stablehlo.multiply %arg0, %1 : tensor<2xi64>
    %3 = stablehlo.divide %arg0, %2 : tensor<2xi64>
    return %3 : tensor<2xi64>
  }
})mlir";

  StablehloModuleBuilder mb;
  {  // Build Main Func
    func::FunctionBuilder fb(mb.get(), "main");
    auto type2xi64 = makeTensorType(mb->getContext(), {2}, ElementType::I64);
    auto arg0 = func::Argument(fb, type2xi64);
    auto cst = Constant(fb, mlir::makeConstant(1L, type2xi64));
    auto add = Add(arg0, cst);
    auto sub = Subtract(arg0, add);
    auto mul = Mul(arg0, sub);
    auto div = Div(arg0, mul);
    func::Return(fb, div);
  }

  // TODO: Make these a FileCheck based test.
  OwningOpRef<ModuleOp> module = mb->build();
  EXPECT_TRUE(succeeded(mlir::verify(*module)));
  EXPECT_EQ(expected, debugString(*module));
}

TEST(MlirBuilderTest, UnaryOps) {
  std::string expected = R"mlir(module {
  func.func @main(%arg0: tensor<2xf32>) -> tensor<2xf32> {
    %0 = stablehlo.abs %arg0 : tensor<2xf32>
    %1 = stablehlo.sine %0 : tensor<2xf32>
    %2 = stablehlo.cosine %1 : tensor<2xf32>
    return %2 : tensor<2xf32>
  }
})mlir";

  StablehloModuleBuilder mb;
  {  // Build Main Func
    func::FunctionBuilder fb(mb.get(), "main");
    auto type2xi64 = makeTensorType(mb->getContext(), {2}, ElementType::F32);
    auto arg0 = func::Argument(fb, type2xi64);
    auto abs = Abs(arg0);
    auto sine = Sine(abs);
    auto cosine = Cosine(sine);
    func::Return(fb, cosine);
  }

  // TODO: Make these a FileCheck based test.
  OwningOpRef<ModuleOp> module = mb->build();
  EXPECT_TRUE(succeeded(mlir::verify(*module)));
  EXPECT_EQ(expected, debugString(*module));
}

TEST(MlirBuilderTest, DotOp) {
  std::string expected = R"mlir(module {
  func.func @main(%arg0: tensor<2x3xf32>, %arg1: tensor<3x2xf32>) -> tensor<2x2xf32> {
    %0 = stablehlo.dot %arg0, %arg1, precision = [HIGHEST, HIGHEST] : (tensor<2x3xf32>, tensor<3x2xf32>) -> tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
  }
})mlir";

  StablehloModuleBuilder mb;
  {  // Build Main Func
    func::FunctionBuilder fb(mb.get(), "main");
    auto& ctx = fb.getContext();
    auto type2x3xi64 = makeTensorType(ctx, {2, 3}, ElementType::F32);
    auto type3x2xi64 = makeTensorType(ctx, {3, 2}, ElementType::F32);
    auto arg0 = func::Argument(fb, type2x3xi64);
    auto arg1 = func::Argument(fb, type3x2xi64);
    auto precision = PrecisionConfigAttr::get(
        &ctx, {Precision::HIGHEST, Precision::HIGHEST});
    auto dot = stablehlo::Dot(arg0, arg1, precision);
    func::Return(fb, dot);
  }

  OwningOpRef<ModuleOp> module = mb->build();
  EXPECT_TRUE(succeeded(mlir::verify(*module)));
  EXPECT_EQ(expected, debugString(*module));
}

TEST(MlirBuilderTest, DotGeneralOp) {
  std::string expected = R"mlir(module {
  func.func @main(%arg0: tensor<2x2x2xi64>, %arg1: tensor<2x2x2xi64>) -> tensor<2x2x2xi64> {
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [HIGHEST, HIGHEST] : (tensor<2x2x2xi64>, tensor<2x2x2xi64>) -> tensor<2x2x2xi64>
    return %0 : tensor<2x2x2xi64>
  }
})mlir";

  StablehloModuleBuilder mb;
  {  // Build Main Func
    func::FunctionBuilder fb(mb.get(), "main");
    auto& ctx = fb.getContext();
    auto type2x2x2xi64 = makeTensorType(ctx, {2, 2, 2}, ElementType::I64);
    auto arg0 = func::Argument(fb, type2x2x2xi64);
    auto arg1 = func::Argument(fb, type2x2x2xi64);

    // TODO(UX): Can we make DotDimensionNumbersAttr have better builders?
    auto dotDimsAttr = DotDimensionNumbersAttr::get(
        &ctx, /*lhsBatchingDimensions=*/{0},
        /*rhsBatchingDimensions=*/{0}, /*lhsContractingDimensions=*/{2},
        /*rhsContractingDimensions=*/{1});

    auto precision = PrecisionConfigAttr::get(
        &ctx, {Precision::HIGHEST, Precision::HIGHEST});
    auto dot = stablehlo::DotGeneral(arg0, arg1, dotDimsAttr, precision);
    func::Return(fb, dot);
  }

  OwningOpRef<ModuleOp> module = mb->build();
  EXPECT_TRUE(succeeded(mlir::verify(*module)));
  EXPECT_EQ(expected, debugString(*module));
}

TEST(MlirBuilderTest, ReduceOp) {
  std::string expected = R"mlir(module {
  func.func @main(%arg0: tensor<2xi64>) -> tensor<i64> {
    %c = stablehlo.constant dense<1> : tensor<i64>
    %0 = stablehlo.reduce(%arg0 init: %c) applies stablehlo.add across dimensions = [0] : (tensor<2xi64>, tensor<i64>) -> tensor<i64>
    return %0 : tensor<i64>
  }
})mlir";

  StablehloModuleBuilder mb;
  {  // Build Main Func
    func::FunctionBuilder fb(mb.get(), "main");
    auto type2xi64 = makeTensorType(mb->getContext(), {2}, ElementType::I64);
    auto typei64 = makeTensorType(mb->getContext(), {}, ElementType::I64);
    auto arg0 = func::Argument(fb, type2xi64);
    auto cst = stablehlo::Constant(fb, mlir::makeConstant(1L, typei64));
    auto reduce = stablehlo::Reduce(
        fb, {arg0}, {cst},
        [&typei64](RegionBuilder& body) {
          buildReduceBody<AddOp>(typei64.getElementType(), body.getRegion(),
                                 body.getOpBuilder());
        },
        /*dimensions=*/{0});
    func::Return(fb, reduce);
  }

  OwningOpRef<ModuleOp> module = mb->build();
  EXPECT_TRUE(succeeded(mlir::verify(*module)));
  EXPECT_EQ(expected, debugString(*module));
}

TEST(MlirBuilderTest, GatherOp) {
  std::string expected = R"mlir(module {
  func.func @main(%arg0: tensor<3xi64>, %arg1: tensor<1x1xi64>) -> tensor<1xi64> {
    %0 = "stablehlo.gather"(%arg0, %arg1) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1>}> : (tensor<3xi64>, tensor<1x1xi64>) -> tensor<1xi64>
    return %0 : tensor<1xi64>
  }
})mlir";

  StablehloModuleBuilder mb;
  {  // Build Main Func
    func::FunctionBuilder fb(mb.get(), "main");
    auto& ctx = fb.getContext();
    auto arg0 = func::Argument(fb, makeTensorType(ctx, {3}, ElementType::I64));
    auto arg1 =
        func::Argument(fb, makeTensorType(ctx, {1, 1}, ElementType::I64));
    // TODO(UX): A bit verbose. Could use a better attr builder function.
    auto gatherDims = GatherDimensionNumbersAttr::get(
        &ctx, /*offset_dims=*/{}, /*collapsed_slice_dims=*/{0},
        /*operandBatchingDims=*/{},
        /*startIndicesBatchingDims=*/{}, /*startIndexMap=*/{0},
        /*index_vector_dim=*/1);
    auto gather =
        stablehlo::Gather(arg0, arg1, gatherDims, /*slice_sizes=*/{1});
    func::Return(fb, gather);
  }

  OwningOpRef<ModuleOp> module = mb->build();
  EXPECT_TRUE(succeeded(mlir::verify(*module)));
  EXPECT_EQ(expected, debugString(*module));
}

TEST(MlirBuilderTest, WhileOp) {
  std::string expected = R"mlir(module {
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
})mlir";

  StablehloModuleBuilder mb;
  {  // Build Main Func
    func::FunctionBuilder fb(mb.get(), "main");
    auto typei64 = makeTensorType(mb->getContext(), {}, ElementType::I64);
    auto arg0 = func::Argument(fb, typei64);
    auto cst = Constant(fb, mlir::makeConstant(1L, typei64));
    auto whl = While(
        fb, arg0,
        [&cst](RegionBuilder& cond) {
          // Note: always use `Arguments(...)` to init block args for WhileOp.
          auto args = Arguments(cond, cond.getOp<WhileOp>());
          auto lt = Compare(args[0], cst, ComparisonDirection::LT);
          return Return(cond, lt);
        },
        [&cst](RegionBuilder& body) {
          auto args = Arguments(body, body.getOp<WhileOp>());
          auto sub1 = Subtract(args[0], cst);
          return Return(body, sub1);
        });
    func::Return(fb, whl);
  }

  OwningOpRef<ModuleOp> module = mb->build();
  EXPECT_TRUE(succeeded(mlir::verify(*module)));
  EXPECT_EQ(expected, debugString(*module));
}

TEST(MlirBuilderTest, ConstantPRED) {
  std::string expected = R"mlir(module {
  func.func @main() -> tensor<i1> {
    %c = stablehlo.constant dense<true> : tensor<i1>
    return %c : tensor<i1>
  }
})mlir";

  StablehloModuleBuilder mb;
  {
    Location funcLoc = fileLineColLoc(mb->getContext(), "main.mlir", 1, 1);
    func::FunctionBuilder fb(mb.get(), "main", funcLoc);
    auto type = makeTensorType(fb.getContext(), {}, ElementType::PRED);
    auto cst = stablehlo::Constant(fb, makeConstant(true, type));
    func::Return(fb, {cst});
  }

  OwningOpRef<ModuleOp> module = mb->build();
  EXPECT_EQ(expected, debugString(*module));
}

TEST(MlirBuilderTest, ConstantI32) {
  std::string expected = R"mlir(module {
  func.func @main() -> tensor<i32> {
    %c = stablehlo.constant dense<1> : tensor<i32>
    return %c : tensor<i32>
  }
})mlir";

  StablehloModuleBuilder mb;
  {
    Location funcLoc = fileLineColLoc(mb->getContext(), "main.mlir", 1, 1);
    func::FunctionBuilder fb(mb.get(), "main", funcLoc);
    auto type = makeTensorType(fb.getContext(), {}, ElementType::I32);
    auto cst = stablehlo::Constant(fb, makeConstant(1, type));
    func::Return(fb, {cst});
  }

  OwningOpRef<ModuleOp> module = mb->build();
  EXPECT_EQ(expected, debugString(*module));
}

TEST(MlirBuilderTest, ConstantF32) {
  std::string expected = R"mlir(module {
  func.func @main() -> tensor<f32> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    return %cst : tensor<f32>
  }
})mlir";

  StablehloModuleBuilder mb;
  {
    Location funcLoc = fileLineColLoc(mb->getContext(), "main.mlir", 1, 1);
    func::FunctionBuilder fb(mb.get(), "main", funcLoc);
    auto type = makeTensorType(fb.getContext(), {}, ElementType::F32);
    auto cst = stablehlo::Constant(fb, makeConstant(1.0, type));
    func::Return(fb, {cst});
  }

  OwningOpRef<ModuleOp> module = mb->build();
  EXPECT_EQ(expected, debugString(*module));
}

TEST(MlirBuilderTest, ConstantComplexF32) {
  std::string expected = R"mlir(module {
  func.func @main() -> tensor<complex<f32>> {
    %cst = stablehlo.constant dense<(1.000000e+00,2.000000e+00)> : tensor<complex<f32>>
    return %cst : tensor<complex<f32>>
  }
})mlir";

  StablehloModuleBuilder mb;
  {
    Location funcLoc = fileLineColLoc(mb->getContext(), "main.mlir", 1, 1);
    func::FunctionBuilder fb(mb.get(), "main", funcLoc);
    auto type = makeTensorType(fb.getContext(), {}, ElementType::COMPLEXF32);
    auto cst = stablehlo::Constant(
        fb, makeConstant(std::complex<double>(1.0, 2.0), type));
    func::Return(fb, {cst});
  }

  OwningOpRef<ModuleOp> module = mb->build();
  EXPECT_EQ(expected, debugString(*module));
}

TEST(MlirBuilderTest, ConstantPredFromInt) {
  std::string expected = R"mlir(module {
  func.func @main() -> tensor<i1> {
    %c = stablehlo.constant dense<true> : tensor<i1>
    return %c : tensor<i1>
  }
})mlir";

  StablehloModuleBuilder mb;
  {
    Location funcLoc = fileLineColLoc(mb->getContext(), "main.mlir", 1, 1);
    func::FunctionBuilder fb(mb.get(), "main", funcLoc);
    auto type = makeTensorType(fb.getContext(), {}, ElementType::PRED);
    auto cst = stablehlo::Constant(fb, makeConstant(1, type));
    func::Return(fb, {cst});
  }

  OwningOpRef<ModuleOp> module = mb->build();
  EXPECT_EQ(expected, debugString(*module));
}

TEST(MlirBuilderTest, ConstantPredFromFloat) {
  std::string expected = R"mlir(module {
  func.func @main() -> tensor<i1> {
    %c = stablehlo.constant dense<true> : tensor<i1>
    return %c : tensor<i1>
  }
})mlir";

  StablehloModuleBuilder mb;
  {
    Location funcLoc = fileLineColLoc(mb->getContext(), "main.mlir", 1, 1);
    func::FunctionBuilder fb(mb.get(), "main", funcLoc);
    auto type = makeTensorType(fb.getContext(), {}, ElementType::PRED);
    auto cst = stablehlo::Constant(fb, makeConstant(1.0, type));
    func::Return(fb, {cst});
  }

  OwningOpRef<ModuleOp> module = mb->build();
  EXPECT_EQ(expected, debugString(*module));
}

TEST(MlirBuilderTest, ConstantPredFromComplexF32) {
  std::string expected = R"mlir(module {
  func.func @main() -> tensor<i1> {
    %c = stablehlo.constant dense<true> : tensor<i1>
    return %c : tensor<i1>
  }
})mlir";

  StablehloModuleBuilder mb;
  {
    Location funcLoc = fileLineColLoc(mb->getContext(), "main.mlir", 1, 1);
    func::FunctionBuilder fb(mb.get(), "main", funcLoc);
    auto type = makeTensorType(fb.getContext(), {}, ElementType::PRED);
    auto cst = stablehlo::Constant(
        fb, makeConstant(std::complex<double>(1.0, 2.0), type));
    func::Return(fb, {cst});
  }

  OwningOpRef<ModuleOp> module = mb->build();
  EXPECT_EQ(expected, debugString(*module));
}

TEST(MlirBuilderTest, ConstantI32FromPred) {
  std::string expected = R"mlir(module {
  func.func @main() -> tensor<i32> {
    %c = stablehlo.constant dense<1> : tensor<i32>
    return %c : tensor<i32>
  }
})mlir";

  StablehloModuleBuilder mb;
  {
    Location funcLoc = fileLineColLoc(mb->getContext(), "main.mlir", 1, 1);
    func::FunctionBuilder fb(mb.get(), "main", funcLoc);
    auto type = makeTensorType(fb.getContext(), {}, ElementType::I32);
    auto cst = stablehlo::Constant(fb, makeConstant(true, type));
    func::Return(fb, {cst});
  }

  OwningOpRef<ModuleOp> module = mb->build();
  EXPECT_EQ(expected, debugString(*module));
}

TEST(MlirBuilderTest, ConstantI32FromFloat) {
  std::string expected = R"mlir(module {
  func.func @main() -> tensor<i32> {
    %c = stablehlo.constant dense<0> : tensor<i32>
    return %c : tensor<i32>
  }
})mlir";

  StablehloModuleBuilder mb;
  {
    Location funcLoc = fileLineColLoc(mb->getContext(), "main.mlir", 1, 1);
    func::FunctionBuilder fb(mb.get(), "main", funcLoc);
    auto type = makeTensorType(fb.getContext(), {}, ElementType::I32);
    auto cst = stablehlo::Constant(fb, makeConstant(0.0, type));
    func::Return(fb, {cst});
  }

  OwningOpRef<ModuleOp> module = mb->build();
  EXPECT_EQ(expected, debugString(*module));
}

TEST(MlirBuilderTest, ConstantI32FromComplexF32) {
  std::string expected = R"mlir(module {
  func.func @main() -> tensor<i32> {
    %c = stablehlo.constant dense<1> : tensor<i32>
    return %c : tensor<i32>
  }
})mlir";

  StablehloModuleBuilder mb;
  {
    Location funcLoc = fileLineColLoc(mb->getContext(), "main.mlir", 1, 1);
    func::FunctionBuilder fb(mb.get(), "main", funcLoc);
    auto type = makeTensorType(fb.getContext(), {}, ElementType::I32);
    auto cst = stablehlo::Constant(
        fb, makeConstant(std::complex<double>(1.0, 2.0), type));
    func::Return(fb, {cst});
  }

  OwningOpRef<ModuleOp> module = mb->build();
  EXPECT_EQ(expected, debugString(*module));
}

TEST(MlirBuilderTest, ConstantF32FromPred) {
  std::string expected = R"mlir(module {
  func.func @main() -> tensor<f32> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    return %cst : tensor<f32>
  }
})mlir";

  StablehloModuleBuilder mb;
  {
    Location funcLoc = fileLineColLoc(mb->getContext(), "main.mlir", 1, 1);
    func::FunctionBuilder fb(mb.get(), "main", funcLoc);
    auto type = makeTensorType(fb.getContext(), {}, ElementType::F32);
    auto cst = stablehlo::Constant(fb, makeConstant(true, type));
    func::Return(fb, {cst});
  }

  OwningOpRef<ModuleOp> module = mb->build();
  EXPECT_EQ(expected, debugString(*module));
}

TEST(MlirBuilderTest, ConstantF32FromInt) {
  std::string expected = R"mlir(module {
  func.func @main() -> tensor<f32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    return %cst : tensor<f32>
  }
})mlir";

  StablehloModuleBuilder mb;
  {
    Location funcLoc = fileLineColLoc(mb->getContext(), "main.mlir", 1, 1);
    func::FunctionBuilder fb(mb.get(), "main", funcLoc);
    auto type = makeTensorType(fb.getContext(), {}, ElementType::F32);
    auto cst = stablehlo::Constant(fb, makeConstant(0, type));
    func::Return(fb, {cst});
  }

  OwningOpRef<ModuleOp> module = mb->build();
  EXPECT_EQ(expected, debugString(*module));
}

TEST(MlirBuilderTest, ConstantF32FromComplexF32) {
  std::string expected = R"mlir(module {
  func.func @main() -> tensor<f32> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    return %cst : tensor<f32>
  }
})mlir";

  StablehloModuleBuilder mb;
  {
    Location funcLoc = fileLineColLoc(mb->getContext(), "main.mlir", 1, 1);
    func::FunctionBuilder fb(mb.get(), "main", funcLoc);
    auto type = makeTensorType(fb.getContext(), {}, ElementType::F32);
    auto cst = stablehlo::Constant(
        fb, makeConstant(std::complex<double>(1.0, 2.0), type));
    func::Return(fb, {cst});
  }

  OwningOpRef<ModuleOp> module = mb->build();
  EXPECT_EQ(expected, debugString(*module));
}

TEST(MlirBuilderTest, ConstantComplexF32FromPred) {
  std::string expected = R"mlir(module {
  func.func @main() -> tensor<complex<f32>> {
    %cst = stablehlo.constant dense<(1.000000e+00,0.000000e+00)> : tensor<complex<f32>>
    return %cst : tensor<complex<f32>>
  }
})mlir";

  StablehloModuleBuilder mb;
  {
    Location funcLoc = fileLineColLoc(mb->getContext(), "main.mlir", 1, 1);
    func::FunctionBuilder fb(mb.get(), "main", funcLoc);
    auto type = makeTensorType(fb.getContext(), {}, ElementType::COMPLEXF32);
    auto cst = stablehlo::Constant(fb, makeConstant(true, type));
    func::Return(fb, {cst});
  }

  OwningOpRef<ModuleOp> module = mb->build();
  EXPECT_EQ(expected, debugString(*module));
}

TEST(MlirBuilderTest, ConstantComplexF32FromInt) {
  std::string expected = R"mlir(module {
  func.func @main() -> tensor<complex<f32>> {
    %cst = stablehlo.constant dense<(1.000000e+00,0.000000e+00)> : tensor<complex<f32>>
    return %cst : tensor<complex<f32>>
  }
})mlir";

  StablehloModuleBuilder mb;
  {
    Location funcLoc = fileLineColLoc(mb->getContext(), "main.mlir", 1, 1);
    func::FunctionBuilder fb(mb.get(), "main", funcLoc);
    auto type = makeTensorType(fb.getContext(), {}, ElementType::COMPLEXF32);
    auto cst = stablehlo::Constant(fb, makeConstant(1, type));
    func::Return(fb, {cst});
  }

  OwningOpRef<ModuleOp> module = mb->build();
  EXPECT_EQ(expected, debugString(*module));
}

TEST(MlirBuilderTest, ConstantComplexF32FromFloat) {
  std::string expected = R"mlir(module {
  func.func @main() -> tensor<complex<f32>> {
    %cst = stablehlo.constant dense<(1.000000e+00,0.000000e+00)> : tensor<complex<f32>>
    return %cst : tensor<complex<f32>>
  }
})mlir";

  StablehloModuleBuilder mb;
  {
    Location funcLoc = fileLineColLoc(mb->getContext(), "main.mlir", 1, 1);
    func::FunctionBuilder fb(mb.get(), "main", funcLoc);
    auto type = makeTensorType(fb.getContext(), {}, ElementType::COMPLEXF32);
    auto cst = stablehlo::Constant(fb, makeConstant(1.0, type));
    func::Return(fb, {cst});
  }

  OwningOpRef<ModuleOp> module = mb->build();
  EXPECT_EQ(expected, debugString(*module));
}

TEST(MlirBuilderTest, ConstantPREDArray) {
  std::string expected = R"mlir(module {
  func.func @main() -> tensor<2xi1> {
    %c = stablehlo.constant dense<[true, false]> : tensor<2xi1>
    return %c : tensor<2xi1>
  }
})mlir";

  StablehloModuleBuilder mb;
  {
    Location funcLoc = fileLineColLoc(mb->getContext(), "main.mlir", 1, 1);
    func::FunctionBuilder fb(mb.get(), "main", funcLoc);
    auto type = makeTensorType(fb.getContext(), {2}, ElementType::PRED);
    auto cst = stablehlo::Constant(
        fb, makeConstant(ArrayRef<bool>({true, false}), type));
    func::Return(fb, {cst});
  }

  OwningOpRef<ModuleOp> module = mb->build();
  EXPECT_EQ(expected, debugString(*module));
}

TEST(MlirBuilderTest, ConstantI2Array) {
  std::string expected = R"mlir(module {
  func.func @main() -> tensor<4xi2> {
    %c = stablehlo.constant dense<[-2, -1, 0, 1]> : tensor<4xi2>
    return %c : tensor<4xi2>
  }
})mlir";

  StablehloModuleBuilder mb;
  {
    Location funcLoc = fileLineColLoc(mb->getContext(), "main.mlir", 1, 1);
    func::FunctionBuilder fb(mb.get(), "main", funcLoc);
    auto type = makeTensorType(fb.getContext(), {4}, ElementType::I2);
    auto cst = stablehlo::Constant(
        fb, makeConstant(ArrayRef<int64_t>({-2, -1, 0, 1}), type));
    func::Return(fb, {cst});
  }

  OwningOpRef<ModuleOp> module = mb->build();
  EXPECT_EQ(expected, debugString(*module));
}

TEST(MlirBuilderTest, ConstantUI2Array) {
  std::string expected = R"mlir(module {
  func.func @main() -> tensor<4xui2> {
    %c = stablehlo.constant dense<[0, 1, 2, 3]> : tensor<4xui2>
    return %c : tensor<4xui2>
  }
})mlir";

  StablehloModuleBuilder mb;
  {
    Location funcLoc = fileLineColLoc(mb->getContext(), "main.mlir", 1, 1);
    func::FunctionBuilder fb(mb.get(), "main", funcLoc);
    auto type = makeTensorType(fb.getContext(), {4}, ElementType::UI2);
    auto cst = stablehlo::Constant(
        fb, makeConstant(ArrayRef<uint64_t>({0, 1, 2, 3}), type));
    func::Return(fb, {cst});
  }

  OwningOpRef<ModuleOp> module = mb->build();
  EXPECT_EQ(expected, debugString(*module));
}

TEST(MlirBuilderTest, ConstantI4Array) {
  std::string expected = R"mlir(module {
  func.func @main() -> tensor<5xi4> {
    %c = stablehlo.constant dense<[-8, -1, 0, 1, 7]> : tensor<5xi4>
    return %c : tensor<5xi4>
  }
})mlir";

  StablehloModuleBuilder mb;
  {
    Location funcLoc = fileLineColLoc(mb->getContext(), "main.mlir", 1, 1);
    func::FunctionBuilder fb(mb.get(), "main", funcLoc);
    auto type = makeTensorType(fb.getContext(), {5}, ElementType::I4);
    auto cst = stablehlo::Constant(
        fb, makeConstant(ArrayRef<int64_t>({-8, -1, 0, 1, 7}), type));
    func::Return(fb, {cst});
  }

  OwningOpRef<ModuleOp> module = mb->build();
  EXPECT_EQ(expected, debugString(*module));
}

TEST(MlirBuilderTest, ConstantUI4Array) {
  std::string expected = R"mlir(module {
  func.func @main() -> tensor<3xui4> {
    %c = stablehlo.constant dense<[0, 8, 15]> : tensor<3xui4>
    return %c : tensor<3xui4>
  }
})mlir";

  StablehloModuleBuilder mb;
  {
    Location funcLoc = fileLineColLoc(mb->getContext(), "main.mlir", 1, 1);
    func::FunctionBuilder fb(mb.get(), "main", funcLoc);
    auto type = makeTensorType(fb.getContext(), {3}, ElementType::UI4);
    auto cst = stablehlo::Constant(
        fb, makeConstant(ArrayRef<uint64_t>({0, 8, 15}), type));
    func::Return(fb, {cst});
  }

  OwningOpRef<ModuleOp> module = mb->build();
  EXPECT_EQ(expected, debugString(*module));
}

TEST(MlirBuilderTest, ConstantI8Array) {
  std::string expected = R"mlir(module {
  func.func @main() -> tensor<5xi8> {
    %c = stablehlo.constant dense<[-128, -9, 0, 8, 127]> : tensor<5xi8>
    return %c : tensor<5xi8>
  }
})mlir";

  StablehloModuleBuilder mb;
  {
    Location funcLoc = fileLineColLoc(mb->getContext(), "main.mlir", 1, 1);
    func::FunctionBuilder fb(mb.get(), "main", funcLoc);
    auto type = makeTensorType(fb.getContext(), {5}, ElementType::I8);
    auto cst = stablehlo::Constant(
        fb, makeConstant(ArrayRef<int8_t>({-128, -9, 0, 8, 127}), type));
    func::Return(fb, {cst});
  }

  OwningOpRef<ModuleOp> module = mb->build();
  EXPECT_EQ(expected, debugString(*module));
}

TEST(MlirBuilderTest, ConstantUI8Array) {
  std::string expected = R"mlir(module {
  func.func @main() -> tensor<3xui8> {
    %c = stablehlo.constant dense<[0, 16, 255]> : tensor<3xui8>
    return %c : tensor<3xui8>
  }
})mlir";

  StablehloModuleBuilder mb;
  {
    Location funcLoc = fileLineColLoc(mb->getContext(), "main.mlir", 1, 1);
    func::FunctionBuilder fb(mb.get(), "main", funcLoc);
    auto type = makeTensorType(fb.getContext(), {3}, ElementType::UI8);
    auto cst = stablehlo::Constant(
        fb, makeConstant(ArrayRef<uint8_t>({0, 16, 255}), type));
    func::Return(fb, {cst});
  }

  OwningOpRef<ModuleOp> module = mb->build();
  EXPECT_EQ(expected, debugString(*module));
}

TEST(MlirBuilderTest, ConstantI16Array) {
  std::string expected = R"mlir(module {
  func.func @main() -> tensor<5xi16> {
    %c = stablehlo.constant dense<[-32768, -129, 0, 128, 32767]> : tensor<5xi16>
    return %c : tensor<5xi16>
  }
})mlir";

  StablehloModuleBuilder mb;
  {
    Location funcLoc = fileLineColLoc(mb->getContext(), "main.mlir", 1, 1);
    func::FunctionBuilder fb(mb.get(), "main", funcLoc);
    auto type = makeTensorType(fb.getContext(), {5}, ElementType::I16);
    auto cst = stablehlo::Constant(
        fb,
        makeConstant(ArrayRef<int64_t>({-32768, -129, 0, 128, 32767}), type));
    func::Return(fb, {cst});
  }

  OwningOpRef<ModuleOp> module = mb->build();
  EXPECT_EQ(expected, debugString(*module));
}

TEST(MlirBuilderTest, ConstantUI16Array) {
  std::string expected = R"mlir(module {
  func.func @main() -> tensor<3xui16> {
    %c = stablehlo.constant dense<[0, 256, 65535]> : tensor<3xui16>
    return %c : tensor<3xui16>
  }
})mlir";

  StablehloModuleBuilder mb;
  {
    Location funcLoc = fileLineColLoc(mb->getContext(), "main.mlir", 1, 1);
    func::FunctionBuilder fb(mb.get(), "main", funcLoc);
    auto type = makeTensorType(fb.getContext(), {3}, ElementType::UI16);
    auto cst = stablehlo::Constant(
        fb, makeConstant(ArrayRef<uint64_t>({0, 256, 65535}), type));
    func::Return(fb, {cst});
  }

  OwningOpRef<ModuleOp> module = mb->build();
  EXPECT_EQ(expected, debugString(*module));
}

TEST(MlirBuilderTest, ConstantI32Array) {
  std::string expected = R"mlir(module {
  func.func @main() -> tensor<5xi32> {
    %c = stablehlo.constant dense<[-2147483648, -65537, 0, 65536, 2147483647]> : tensor<5xi32>
    return %c : tensor<5xi32>
  }
})mlir";

  StablehloModuleBuilder mb;
  {
    Location funcLoc = fileLineColLoc(mb->getContext(), "main.mlir", 1, 1);
    func::FunctionBuilder fb(mb.get(), "main", funcLoc);
    auto type = makeTensorType(fb.getContext(), {5}, ElementType::I32);
    auto cst = stablehlo::Constant(
        fb, makeConstant(
                ArrayRef<int64_t>({-2147483648, -65537, 0, 65536, 2147483647}),
                type));
    func::Return(fb, {cst});
  }

  OwningOpRef<ModuleOp> module = mb->build();
  EXPECT_EQ(expected, debugString(*module));
}

TEST(MlirBuilderTest, ConstantUI32Array) {
  std::string expected = R"mlir(module {
  func.func @main() -> tensor<3xui32> {
    %c = stablehlo.constant dense<[0, 65536, 4294967295]> : tensor<3xui32>
    return %c : tensor<3xui32>
  }
})mlir";

  StablehloModuleBuilder mb;
  {
    Location funcLoc = fileLineColLoc(mb->getContext(), "main.mlir", 1, 1);
    func::FunctionBuilder fb(mb.get(), "main", funcLoc);
    auto type = makeTensorType(fb.getContext(), {3}, ElementType::UI32);
    auto cst = stablehlo::Constant(
        fb, makeConstant(ArrayRef<uint64_t>({0, 65536, 4294967295}), type));
    func::Return(fb, {cst});
  }

  OwningOpRef<ModuleOp> module = mb->build();
  EXPECT_EQ(expected, debugString(*module));
}

TEST(MlirBuilderTest, ConstantI64Array) {
  std::string expected = R"mlir(module {
  func.func @main() -> tensor<5xi64> {
    %c = stablehlo.constant dense<[-9223372036854775808, -2147483649, 0, 2147483648, 9223372036854775807]> : tensor<5xi64>
    return %c : tensor<5xi64>
  }
})mlir";

  StablehloModuleBuilder mb;
  {
    Location funcLoc = fileLineColLoc(mb->getContext(), "main.mlir", 1, 1);
    func::FunctionBuilder fb(mb.get(), "main", funcLoc);
    auto type = makeTensorType(fb.getContext(), {5}, ElementType::I64);
    auto cst = stablehlo::Constant(
        fb,
        makeConstant(ArrayRef<int64_t>({-9223372036854775807 - 1, -2147483649,
                                        0, 2147483648, 9223372036854775807}),
                     type));
    func::Return(fb, {cst});
  }

  OwningOpRef<ModuleOp> module = mb->build();
  EXPECT_EQ(expected, debugString(*module));
}

TEST(MlirBuilderTest, ConstantUI64Array) {
  std::string expected = R"mlir(module {
  func.func @main() -> tensor<3xui64> {
    %c = stablehlo.constant dense<[0, 4294967296, 18446744073709551615]> : tensor<3xui64>
    return %c : tensor<3xui64>
  }
})mlir";

  StablehloModuleBuilder mb;
  {
    Location funcLoc = fileLineColLoc(mb->getContext(), "main.mlir", 1, 1);
    func::FunctionBuilder fb(mb.get(), "main", funcLoc);
    auto type = makeTensorType(fb.getContext(), {3}, ElementType::UI64);
    auto cst = stablehlo::Constant(
        fb,
        makeConstant(
            ArrayRef<uint64_t>({0, 4294967296, 18446744073709551615UL}), type));
    func::Return(fb, {cst});
  }

  OwningOpRef<ModuleOp> module = mb->build();
  EXPECT_EQ(expected, debugString(*module));
}

TEST(MlirBuilderTest, ConstantF4E2M1FNArray) {
  std::string expected = R"mlir(module {
  func.func @main() -> tensor<10xf4E2M1FN> {
    %cst = stablehlo.constant dense<[0.000000e+00, -0.000000e+00, 1.000000e+00, 0.000000e+00, 0.000000e+00, 3.000000e+00, 6.000000e+00, 6.000000e+00, 1.000000e+00, 6.000000e+00]> : tensor<10xf4E2M1FN>
    return %cst : tensor<10xf4E2M1FN>
  }
})mlir";

  StablehloModuleBuilder mb;
  {
    Location funcLoc = fileLineColLoc(mb->getContext(), "main.mlir", 1, 1);
    func::FunctionBuilder fb(mb.get(), "main", funcLoc);
    auto type = makeTensorType(fb.getContext(), {10}, ElementType::F4E2M1FN);
    auto cst = stablehlo::Constant(
        fb, makeConstant(ArrayRef<double>({0.0, -0.0, 1.0, 0.125, 0.1, 3.1415,
                                           0x07, 0x0F, 0x01, 0x09}),
                         type));
    func::Return(fb, {cst});
  }

  OwningOpRef<ModuleOp> module = mb->build();
  EXPECT_EQ(expected, debugString(*module));
}

TEST(MlirBuilderTest, ConstantF6E2M3FNArray) {
  std::string expected = R"mlir(module {
  func.func @main() -> tensor<10xf6E2M3FN> {
    %cst = stablehlo.constant dense<[0.000000e+00, -0.000000e+00, 1.000000e+00, 1.250000e-01, 1.250000e-01, 3.250000e+00, 7.500000e+00, 7.500000e+00, 1.000000e+00, 7.500000e+00]> : tensor<10xf6E2M3FN>
    return %cst : tensor<10xf6E2M3FN>
  }
})mlir";

  StablehloModuleBuilder mb;
  {
    Location funcLoc = fileLineColLoc(mb->getContext(), "main.mlir", 1, 1);
    func::FunctionBuilder fb(mb.get(), "main", funcLoc);
    auto type = makeTensorType(fb.getContext(), {10}, ElementType::F6E2M3FN);
    auto cst = stablehlo::Constant(
        fb, makeConstant(ArrayRef<double>({0.0, -0.0, 1.0, 0.125, 0.1, 3.1415,
                                           0x1F, 0x3F, 0x01, 0x21}),
                         type));
    func::Return(fb, {cst});
  }

  OwningOpRef<ModuleOp> module = mb->build();
  EXPECT_EQ(expected, debugString(*module));
}

TEST(MlirBuilderTest, ConstantF6E3M2FNArray) {
  std::string expected = R"mlir(module {
  func.func @main() -> tensor<10xf6E3M2FN> {
    %cst = stablehlo.constant dense<[0.000000e+00, -0.000000e+00, 1.000000e+00, 1.250000e-01, 1.250000e-01, 3.000000e+00, 2.800000e+01, 2.800000e+01, 1.000000e+00, 2.800000e+01]> : tensor<10xf6E3M2FN>
    return %cst : tensor<10xf6E3M2FN>
  }
})mlir";

  StablehloModuleBuilder mb;
  {
    Location funcLoc = fileLineColLoc(mb->getContext(), "main.mlir", 1, 1);
    func::FunctionBuilder fb(mb.get(), "main", funcLoc);
    auto type = makeTensorType(fb.getContext(), {10}, ElementType::F6E3M2FN);
    auto cst = stablehlo::Constant(
        fb, makeConstant(ArrayRef<double>({0.0, -0.0, 1.0, 0.125, 0.1, 3.1415,
                                           0x1F, 0x3F, 0x01, 0x21}),
                         type));
    func::Return(fb, {cst});
  }

  OwningOpRef<ModuleOp> module = mb->build();
  EXPECT_EQ(expected, debugString(*module));
}

TEST(MlirBuilderTest, ConstantF8E3M4Array) {
  std::string expected = R"mlir(module {
  func.func @main() -> tensor<10xf8E3M4> {
    %cst = stablehlo.constant dense<[0.000000e+00, -0.000000e+00, 1.000000e+00, 1.250000e-01, 9.375000e-02, 3.125000e+00, 0x70, 0x70, 1.000000e+00, 0x70]> : tensor<10xf8E3M4>
    return %cst : tensor<10xf8E3M4>
  }
})mlir";

  StablehloModuleBuilder mb;
  {
    Location funcLoc = fileLineColLoc(mb->getContext(), "main.mlir", 1, 1);
    func::FunctionBuilder fb(mb.get(), "main", funcLoc);
    auto type = makeTensorType(fb.getContext(), {10}, ElementType::F8E3M4);
    auto cst = stablehlo::Constant(
        fb, makeConstant(ArrayRef<double>({0.0, -0.0, 1.0, 0.125, 0.1, 3.1415,
                                           0x7F, 0xFF, 0x01, 0x81}),
                         type));
    func::Return(fb, {cst});
  }

  OwningOpRef<ModuleOp> module = mb->build();
  EXPECT_EQ(expected, debugString(*module));
}

TEST(MlirBuilderTest, ConstantF8E4M3B11FNUZArray) {
  std::string expected = R"mlir(module {
  func.func @main() -> tensor<10xf8E4M3B11FNUZ> {
    %cst = stablehlo.constant dense<[0.000000e+00, 0.000000e+00, 1.000000e+00, 1.250000e-01, 1.015630e-01, 3.250000e+00, 0x80, 0x80, 1.000000e+00, 0x80]> : tensor<10xf8E4M3B11FNUZ>
    return %cst : tensor<10xf8E4M3B11FNUZ>
  }
})mlir";

  StablehloModuleBuilder mb;
  {
    Location funcLoc = fileLineColLoc(mb->getContext(), "main.mlir", 1, 1);
    func::FunctionBuilder fb(mb.get(), "main", funcLoc);
    auto type =
        makeTensorType(fb.getContext(), {10}, ElementType::F8E4M3B11FNUZ);
    auto cst = stablehlo::Constant(
        fb, makeConstant(ArrayRef<double>({0.0, -0.0, 1.0, 0.125, 0.1, 3.1415,
                                           0x7F, 0xFF, 0x01, 0x81}),
                         type));
    func::Return(fb, {cst});
  }

  OwningOpRef<ModuleOp> module = mb->build();
  EXPECT_EQ(expected, debugString(*module));
}

TEST(MlirBuilderTest, ConstantF8E4M3Array) {
  std::string expected = R"mlir(module {
  func.func @main() -> tensor<10xf8E4M3> {
    %cst = stablehlo.constant dense<[0.000000e+00, -0.000000e+00, 1.000000e+00, 1.250000e-01, 1.015630e-01, 3.250000e+00, 1.280000e+02, 0x78, 1.000000e+00, 1.280000e+02]> : tensor<10xf8E4M3>
    return %cst : tensor<10xf8E4M3>
  }
})mlir";

  StablehloModuleBuilder mb;
  {
    Location funcLoc = fileLineColLoc(mb->getContext(), "main.mlir", 1, 1);
    func::FunctionBuilder fb(mb.get(), "main", funcLoc);
    auto type = makeTensorType(fb.getContext(), {10}, ElementType::F8E4M3);
    auto cst = stablehlo::Constant(
        fb, makeConstant(ArrayRef<double>({0.0, -0.0, 1.0, 0.125, 0.1, 3.1415,
                                           0x7F, 0xFF, 0x01, 0x81}),
                         type));
    func::Return(fb, {cst});
  }

  OwningOpRef<ModuleOp> module = mb->build();
  EXPECT_EQ(expected, debugString(*module));
}

TEST(MlirBuilderTest, ConstantF8E4M3FNArray) {
  std::string expected = R"mlir(module {
  func.func @main() -> tensor<10xf8E4M3FN> {
    %cst = stablehlo.constant dense<[0.000000e+00, -0.000000e+00, 1.000000e+00, 1.250000e-01, 1.015630e-01, 3.250000e+00, 1.280000e+02, 2.560000e+02, 1.000000e+00, 1.280000e+02]> : tensor<10xf8E4M3FN>
    return %cst : tensor<10xf8E4M3FN>
  }
})mlir";

  StablehloModuleBuilder mb;
  {
    Location funcLoc = fileLineColLoc(mb->getContext(), "main.mlir", 1, 1);
    func::FunctionBuilder fb(mb.get(), "main", funcLoc);
    auto type = makeTensorType(fb.getContext(), {10}, ElementType::F8E4M3FN);
    auto cst = stablehlo::Constant(
        fb, makeConstant(ArrayRef<double>({0.0, -0.0, 1.0, 0.125, 0.1, 3.1415,
                                           0x7F, 0xFF, 0x01, 0x81}),
                         type));
    func::Return(fb, {cst});
  }

  OwningOpRef<ModuleOp> module = mb->build();
  EXPECT_EQ(expected, debugString(*module));
}

TEST(MlirBuilderTest, ConstantF8E4M3FNUZArray) {
  std::string expected = R"mlir(module {
  func.func @main() -> tensor<10xf8E4M3FNUZ> {
    %cst = stablehlo.constant dense<[0.000000e+00, 0.000000e+00, 1.000000e+00, 1.250000e-01, 1.015630e-01, 3.250000e+00, 1.280000e+02, 0x80, 1.000000e+00, 1.280000e+02]> : tensor<10xf8E4M3FNUZ>
    return %cst : tensor<10xf8E4M3FNUZ>
  }
})mlir";

  StablehloModuleBuilder mb;
  {
    Location funcLoc = fileLineColLoc(mb->getContext(), "main.mlir", 1, 1);
    func::FunctionBuilder fb(mb.get(), "main", funcLoc);
    auto type = makeTensorType(fb.getContext(), {10}, ElementType::F8E4M3FNUZ);
    auto cst = stablehlo::Constant(
        fb, makeConstant(ArrayRef<double>({0.0, -0.0, 1.0, 0.125, 0.1, 3.1415,
                                           0x7F, 0xFF, 0x01, 0x81}),
                         type));
    func::Return(fb, {cst});
  }

  OwningOpRef<ModuleOp> module = mb->build();
  EXPECT_EQ(expected, debugString(*module));
}

TEST(MlirBuilderTest, ConstantF8E5M2Array) {
  std::string expected = R"mlir(module {
  func.func @main() -> tensor<10xf8E5M2> {
    %cst = stablehlo.constant dense<[0.000000e+00, -0.000000e+00, 1.000000e+00, 1.250000e-01, 9.375000e-02, 3.000000e+00, 1.280000e+02, 2.560000e+02, 1.000000e+00, 1.280000e+02]> : tensor<10xf8E5M2>
    return %cst : tensor<10xf8E5M2>
  }
})mlir";

  StablehloModuleBuilder mb;
  {
    Location funcLoc = fileLineColLoc(mb->getContext(), "main.mlir", 1, 1);
    func::FunctionBuilder fb(mb.get(), "main", funcLoc);
    auto type = makeTensorType(fb.getContext(), {10}, ElementType::F8E5M2);
    auto cst = stablehlo::Constant(
        fb, makeConstant(ArrayRef<double>({0.0, -0.0, 1.0, 0.125, 0.1, 3.1415,
                                           0x7F, 0xFF, 0x01, 0x81}),
                         type));
    func::Return(fb, {cst});
  }

  OwningOpRef<ModuleOp> module = mb->build();
  EXPECT_EQ(expected, debugString(*module));
}

TEST(MlirBuilderTest, ConstantF8E5M2FNUZArray) {
  std::string expected = R"mlir(module {
  func.func @main() -> tensor<10xf8E5M2FNUZ> {
    %cst = stablehlo.constant dense<[0.000000e+00, 0.000000e+00, 1.000000e+00, 1.250000e-01, 9.375000e-02, 3.000000e+00, 1.280000e+02, 2.560000e+02, 1.000000e+00, 1.280000e+02]> : tensor<10xf8E5M2FNUZ>
    return %cst : tensor<10xf8E5M2FNUZ>
  }
})mlir";

  StablehloModuleBuilder mb;
  {
    Location funcLoc = fileLineColLoc(mb->getContext(), "main.mlir", 1, 1);
    func::FunctionBuilder fb(mb.get(), "main", funcLoc);
    auto type = makeTensorType(fb.getContext(), {10}, ElementType::F8E5M2FNUZ);
    auto cst = stablehlo::Constant(
        fb, makeConstant(ArrayRef<double>({0.0, -0.0, 1.0, 0.125, 0.1, 3.1415,
                                           0x7F, 0xFF, 0x01, 0x81}),
                         type));
    func::Return(fb, {cst});
  }

  OwningOpRef<ModuleOp> module = mb->build();
  EXPECT_EQ(expected, debugString(*module));
}

TEST(MlirBuilderTest, ConstantF8E8M0FNUArray) {
  std::string expected = R"mlir(module {
  func.func @main() -> tensor<8xf8E8M0FNU> {
    %cst = stablehlo.constant dense<[5.877470e-39, 1.000000e+00, 1.250000e-01, 1.250000e-01, 4.000000e+00, 5.877470e-39, 1.280000e+02, 2.560000e+02]> : tensor<8xf8E8M0FNU>
    return %cst : tensor<8xf8E8M0FNU>
  }
})mlir";

  StablehloModuleBuilder mb;
  {
    Location funcLoc = fileLineColLoc(mb->getContext(), "main.mlir", 1, 1);
    func::FunctionBuilder fb(mb.get(), "main", funcLoc);
    auto type = makeTensorType(fb.getContext(), {8}, ElementType::F8E8M0FNU);
    auto cst = stablehlo::Constant(
        fb, makeConstant(ArrayRef<double>(
                             {0.0, 1.0, 0.125, 0.1, 3.1415, 0x00, 0x80, 0xFF}),
                         type));
    func::Return(fb, {cst});
  }

  OwningOpRef<ModuleOp> module = mb->build();
  EXPECT_EQ(expected, debugString(*module));
}

TEST(MlirBuilderTest, ConstantBF16Array) {
  std::string expected = R"mlir(module {
  func.func @main() -> tensor<11xbf16> {
    %cst = stablehlo.constant dense<[0.000000e+00, -0.000000e+00, 1.000000e+00, 1.250000e-01, 1.000980e-01, 3.140630e+00, 3.264000e+04, 6.553600e+04, 3.276800e+04, 1.000000e+00, 3.276800e+04]> : tensor<11xbf16>
    return %cst : tensor<11xbf16>
  }
})mlir";

  StablehloModuleBuilder mb;
  {
    Location funcLoc = fileLineColLoc(mb->getContext(), "main.mlir", 1, 1);
    func::FunctionBuilder fb(mb.get(), "main", funcLoc);
    auto type = makeTensorType(fb.getContext(), {11}, ElementType::BF16);
    auto cst = stablehlo::Constant(
        fb,
        makeConstant(ArrayRef<double>({0.0, -0.0, 1.0, 0.125, 0.1, 3.140630,
                                       0x7F80, 0xFF80, 0x7FFF, 0x0001, 0x8001}),
                     type));
    func::Return(fb, {cst});
  }

  OwningOpRef<ModuleOp> module = mb->build();
  EXPECT_EQ(expected, debugString(*module));
}

TEST(MlirBuilderTest, ConstantF16Array) {
  std::string expected = R"mlir(module {
  func.func @main() -> tensor<11xf16> {
    %cst = stablehlo.constant dense<[0.000000e+00, -0.000000e+00, 1.000000e+00, 1.250000e-01, 9.997550e-02, 3.140630e+00, 3.174400e+04, 6.451200e+04, 3.276800e+04, 1.000000e+00, 3.276800e+04]> : tensor<11xf16>
    return %cst : tensor<11xf16>
  }
})mlir";

  StablehloModuleBuilder mb;
  {
    Location funcLoc = fileLineColLoc(mb->getContext(), "main.mlir", 1, 1);
    func::FunctionBuilder fb(mb.get(), "main", funcLoc);
    auto type = makeTensorType(fb.getContext(), {11}, ElementType::F16);
    auto cst = stablehlo::Constant(
        fb,
        makeConstant(ArrayRef<double>({0.0, -0.0, 1.0, 0.125, 0.1, 3.140630,
                                       0x7C00, 0xFC00, 0x7FFF, 0x0001, 0x8001}),
                     type));
    func::Return(fb, {cst});
  }

  OwningOpRef<ModuleOp> module = mb->build();
  EXPECT_EQ(expected, debugString(*module));
}

TEST(MlirBuilderTest, ConstantF32Array) {
  std::string expected = R"mlir(module {
  func.func @main() -> tensor<11xf32> {
    %cst = stablehlo.constant dense<[0.000000e+00, -0.000000e+00, 1.000000e+00, 1.250000e-01, 1.000000e-01, 3.14159274, 2.13909504E+9, 4.28657869E+9, 2.14748365E+9, 1.000000e+00, 2.14748365E+9]> : tensor<11xf32>
    return %cst : tensor<11xf32>
  }
})mlir";

  StablehloModuleBuilder mb;
  {
    Location funcLoc = fileLineColLoc(mb->getContext(), "main.mlir", 1, 1);
    func::FunctionBuilder fb(mb.get(), "main", funcLoc);
    auto type = makeTensorType(fb.getContext(), {11}, ElementType::F32);
    auto cst = stablehlo::Constant(
        fb, makeConstant(ArrayRef<double>({0.0, -0.0, 1.0, 0.125, 0.1,
                                           3.14159274, 0x7F800000, 0xFF800000,
                                           0x7FFFFFFF, 0x00000001, 0x80000001}),
                         type));
    func::Return(fb, {cst});
  }

  OwningOpRef<ModuleOp> module = mb->build();
  EXPECT_EQ(expected, debugString(*module));
}

TEST(MlirBuilderTest, ConstantF64Array) {
  std::string expected = R"mlir(module {
  func.func @main() -> tensor<11xf64> {
    %cst = stablehlo.constant dense<[0.000000e+00, -0.000000e+00, 1.000000e+00, 1.250000e-01, 1.000000e-01, 3.1415926535897931, 9.2188684372274053E+18, 1.8442240474082181E+19, 9.2233720368547758E+18, 1.000000e+00, 9.2233720368547758E+18]> : tensor<11xf64>
    return %cst : tensor<11xf64>
  }
})mlir";

  StablehloModuleBuilder mb;
  {
    Location funcLoc = fileLineColLoc(mb->getContext(), "main.mlir", 1, 1);
    func::FunctionBuilder fb(mb.get(), "main", funcLoc);
    auto type = makeTensorType(fb.getContext(), {11}, ElementType::F64);
    auto cst = stablehlo::Constant(
        fb, makeConstant(
                ArrayRef<long double>({0.0, -0.0, 1.0, 0.125, 0.1,
                                       3.1415926535897931, 0x7FF0000000000000,
                                       0xFFF0000000000000, 0x7FFFFFFFFFFFFFFF,
                                       0x0000000000000001, 0x8000000000000001}),
                type));
    func::Return(fb, {cst});
  }

  OwningOpRef<ModuleOp> module = mb->build();
  EXPECT_EQ(expected, debugString(*module));
}

TEST(MlirBuilderTest, ConstantComplexF32Array) {
  std::string expected = R"mlir(module {
  func.func @main() -> tensor<2xcomplex<f32>> {
    %cst = stablehlo.constant dense<[(1.500000e+00,2.500000e+00), (3.500000e+00,4.500000e+00)]> : tensor<2xcomplex<f32>>
    return %cst : tensor<2xcomplex<f32>>
  }
})mlir";

  StablehloModuleBuilder mb;
  {
    Location funcLoc = fileLineColLoc(mb->getContext(), "main.mlir", 1, 1);
    func::FunctionBuilder fb(mb.get(), "main", funcLoc);
    auto type = makeTensorType(fb.getContext(), {2}, ElementType::COMPLEXF32);
    auto cst = stablehlo::Constant(
        fb,
        makeConstant(ArrayRef<std::complex<double>>({{1.5, 2.5}, {3.5, 4.5}}),
                     type));
    func::Return(fb, {cst});
  }

  OwningOpRef<ModuleOp> module = mb->build();
  EXPECT_EQ(expected, debugString(*module));
}

TEST(MlirBuilderTest, ConstantComplexF64Array) {
  std::string expected = R"mlir(module {
  func.func @main() -> tensor<2xcomplex<f64>> {
    %cst = stablehlo.constant dense<[(1.500000e+00,2.500000e+00), (3.500000e+00,4.500000e+00)]> : tensor<2xcomplex<f64>>
    return %cst : tensor<2xcomplex<f64>>
  }
})mlir";

  StablehloModuleBuilder mb;
  {
    Location funcLoc = fileLineColLoc(mb->getContext(), "main.mlir", 1, 1);
    func::FunctionBuilder fb(mb.get(), "main", funcLoc);
    auto type = makeTensorType(fb.getContext(), {2}, ElementType::COMPLEXF64);
    auto cst = stablehlo::Constant(
        fb,
        makeConstant(ArrayRef<std::complex<double>>({{1.5, 2.5}, {3.5, 4.5}}),
                     type));
    func::Return(fb, {cst});
  }

  OwningOpRef<ModuleOp> module = mb->build();
  EXPECT_EQ(expected, debugString(*module));
}

TEST(MlirBuilderTest, ConvertElementTypeF32ToI32) {
  std::string expected = R"mlir(module {
  func.func @main() -> tensor<2xi32> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<2xf32>
    %0 = stablehlo.convert %cst : (tensor<2xf32>) -> tensor<2xi32>
    return %0 : tensor<2xi32>
  }
})mlir";

  StablehloModuleBuilder mb;
  {
    Location funcLoc = fileLineColLoc(mb->getContext(), "main.mlir", 1, 1);
    func::FunctionBuilder fb(mb.get(), "main", funcLoc);
    auto& ctx = fb.getContext();
    auto type = makeTensorType(ctx, {2}, ElementType::F32);
    auto cst = stablehlo::Constant(fb, makeConstant(1.0, type));
    auto converted = stablehlo::ConvertElementType(cst, ElementType::I32);
    func::Return(fb, converted);
  }

  OwningOpRef<ModuleOp> module = mb->build();
  EXPECT_EQ(expected, debugString(*module));
}

TEST(MlirBuilderTest, ConvertElementTypeComplexToReal) {
  std::string expected = R"mlir(module {
  func.func @main() -> tensor<2xf32> {
    %cst = stablehlo.constant dense<(1.000000e+00,0.000000e+00)> : tensor<2xcomplex<f32>>
    %0 = stablehlo.real %cst : (tensor<2xcomplex<f32>>) -> tensor<2xf32>
    %1 = stablehlo.convert %0 : tensor<2xf32>
    return %1 : tensor<2xf32>
  }
})mlir";

  StablehloModuleBuilder mb;
  {
    Location funcLoc = fileLineColLoc(mb->getContext(), "main.mlir", 1, 1);
    func::FunctionBuilder fb(mb.get(), "main", funcLoc);
    auto& ctx = fb.getContext();
    auto type = makeTensorType(ctx, {2}, ElementType::COMPLEXF32);
    auto cst = stablehlo::Constant(fb, makeConstant(1.0, type));
    auto converted = stablehlo::ConvertElementType(cst, ElementType::F32);
    func::Return(fb, converted);
  }

  OwningOpRef<ModuleOp> module = mb->build();
  EXPECT_EQ(expected, debugString(*module));
}

TEST(MlirBuilderTest, ConstantI64SmallVector) {
  std::string expected = R"mlir(module {
  func.func @main() -> tensor<8xi64> {
    %c = stablehlo.constant dense<[0, 1, 2, 3, 4, 5, 6, 7]> : tensor<8xi64>
    return %c : tensor<8xi64>
  }
})mlir";

  StablehloModuleBuilder mb;
  {
    Location funcLoc = fileLineColLoc(mb->getContext(), "main.mlir", 1, 1);
    func::FunctionBuilder fb(mb.get(), "main", funcLoc);
    auto type = makeTensorType(fb.getContext(), {8}, ElementType::I64);
    auto cst = stablehlo::Constant(
        fb, makeConstant(SmallVector<int64_t>{0, 1, 2, 3, 4, 5, 6, 7}, type));
    func::Return(fb, {cst});
  }

  OwningOpRef<ModuleOp> module = mb->build();
  EXPECT_EQ(expected, debugString(*module));
}

TEST(MlirBuilderTest, ConstantI64Vector) {
  std::string expected = R"mlir(module {
  func.func @main() -> tensor<8xi64> {
    %c = stablehlo.constant dense<[0, 1, 2, 3, 4, 5, 6, 7]> : tensor<8xi64>
    return %c : tensor<8xi64>
  }
})mlir";

  StablehloModuleBuilder mb;
  {
    Location funcLoc = fileLineColLoc(mb->getContext(), "main.mlir", 1, 1);
    func::FunctionBuilder fb(mb.get(), "main", funcLoc);
    auto type = makeTensorType(fb.getContext(), {8}, ElementType::I64);
    auto cst = stablehlo::Constant(
        fb, makeConstant(std::vector<int64_t>{0, 1, 2, 3, 4, 5, 6, 7}, type));
    func::Return(fb, {cst});
  }

  OwningOpRef<ModuleOp> module = mb->build();
  EXPECT_EQ(expected, debugString(*module));
}

TEST(MlirBuilderTest, EmptyConstantI64) {
  std::string expected = R"mlir(module {
  func.func @main() -> tensor<0xi64> {
    %c = stablehlo.constant dense<> : tensor<0xi64>
    return %c : tensor<0xi64>
  }
})mlir";

  StablehloModuleBuilder mb;
  {
    Location funcLoc = fileLineColLoc(mb->getContext(), "main.mlir", 1, 1);
    func::FunctionBuilder fb(mb.get(), "main", funcLoc);
    auto type = makeTensorType(fb.getContext(), {0}, ElementType::I64);
    auto cst = stablehlo::Constant(fb, makeConstant(ArrayRef<int64_t>{}, type));
    func::Return(fb, {cst});
  }

  OwningOpRef<ModuleOp> module = mb->build();
  EXPECT_EQ(expected, debugString(*module));
}

TEST(MlirBuilderTest, EmptyConstantMismatchedTypeI64) {
  std::string expected = R"mlir(module {
  func.func @main() -> tensor<0xi64> {
    %c = stablehlo.constant dense<> : tensor<0xi64>
    return %c : tensor<0xi64>
  }
})mlir";

  StablehloModuleBuilder mb;
  {
    Location funcLoc = fileLineColLoc(mb->getContext(), "main.mlir", 1, 1);
    func::FunctionBuilder fb(mb.get(), "main", funcLoc);
    auto type = makeTensorType(fb.getContext(), {0}, ElementType::I64);
    // Pass double data with i64 type.
    auto cst = stablehlo::Constant(fb, makeConstant(ArrayRef<double>{}, type));
    func::Return(fb, {cst});
  }

  OwningOpRef<ModuleOp> module = mb->build();
  EXPECT_EQ(expected, debugString(*module));
}

TEST(MlirBuilderTest, EmptyConstantMismatchedTypeAPIntF64) {
  std::string expected = R"mlir(module {
  func.func @main() -> tensor<0xf64> {
    %cst = stablehlo.constant dense<> : tensor<0xf64>
    return %cst : tensor<0xf64>
  }
})mlir";

  StablehloModuleBuilder mb;
  {
    Location funcLoc = fileLineColLoc(mb->getContext(), "main.mlir", 1, 1);
    func::FunctionBuilder fb(mb.get(), "main", funcLoc);
    auto type = makeTensorType(fb.getContext(), {0}, ElementType::F64);
    // Pass double data with i64 type.
    auto cst = stablehlo::Constant(fb, makeConstant(ArrayRef<int64_t>{}, type));
    func::Return(fb, {cst});
  }

  OwningOpRef<ModuleOp> module = mb->build();
  EXPECT_EQ(expected, debugString(*module));
}

TEST(MlirBuilderTest, VariadicResult) {
  std::string expected = R"mlir(module {
  func.func @main() -> (tensor<f64>, tensor<f64>) {
    %0:2 = stablehlo.custom_call @two_outs() : () -> (tensor<f64>, tensor<f64>)
    return %0#0, %0#1 : tensor<f64>, tensor<f64>
  }
})mlir";

  StablehloModuleBuilder mb;
  {
    Location funcLoc = fileLineColLoc(mb->getContext(), "main.mlir", 1, 1);
    func::FunctionBuilder fb(mb.get(), "main", funcLoc);
    auto type = makeTensorType(fb.getContext(), {}, ElementType::F64);
    SmallVector<Type> resultTypes = {type, type};
    // Pass double data with i64 type.
    auto cc = stablehlo::CustomCall(fb, resultTypes, {}, "two_outs");
    func::Return(fb, {cc});
  }

  OwningOpRef<ModuleOp> module = mb->build();
  EXPECT_EQ(expected, debugString(*module));
}

////////
// Custom Attribute Tests
////////

TEST(MlirBuilderTest, ResultAccuracyAttrDefault) {
  std::string expected = R"mlir(module {
  func.func @main(%arg0: tensor<2xf32>) -> tensor<2xf32> {
    %0 = stablehlo.exponential %arg0 : tensor<2xf32>
    return %0 : tensor<2xf32>
  }
})mlir";

  StablehloModuleBuilder mb;
  {
    Location funcLoc = fileLineColLoc(mb->getContext(), "main.mlir", 1, 1);
    func::FunctionBuilder fb(mb.get(), "main", funcLoc);
    auto type = makeTensorType(fb.getContext(), {2}, ElementType::F32);
    auto arg0 = func::Argument(fb, type);
    auto exp = Exp(arg0);
    func::Return(fb, {exp});
  }

  OwningOpRef<ModuleOp> module = mb->build();
  EXPECT_EQ(expected, debugString(*module));
}

TEST(MlirBuilderTest, ResultAccuracyAttrHighest) {
  std::string expected = R"mlir(module {
  func.func @main(%arg0: tensor<2xf32>) -> tensor<2xf32> {
    %0 = stablehlo.exponential %arg0 {result_accuracy = #stablehlo.result_accuracy<mode = #stablehlo.result_accuracy_mode<HIGHEST>>} : tensor<2xf32>
    return %0 : tensor<2xf32>
  }
})mlir";

  StablehloModuleBuilder mb;
  {
    Location funcLoc = fileLineColLoc(mb->getContext(), "main.mlir", 1, 1);
    func::FunctionBuilder fb(mb.get(), "main", funcLoc);
    auto type = makeTensorType(fb.getContext(), {2}, ElementType::F32);
    auto arg0 = func::Argument(fb, type);
    auto resultAccuracy =
        ResultAccuracyAttr::get(&fb.getContext(), ResultAccuracyMode::HIGHEST);
    auto exp = Exp(arg0, resultAccuracy);
    func::Return(fb, {exp});
  }

  OwningOpRef<ModuleOp> module = mb->build();
  EXPECT_EQ(expected, debugString(*module));
}

TEST(MlirBuilderTest, ResultAccuracyAttrTolerance) {
  std::string expected = R"mlir(module {
  func.func @main(%arg0: tensor<2xf32>) -> tensor<2xf32> {
    %0 = stablehlo.exponential %arg0 {result_accuracy = #stablehlo.result_accuracy<atol = 1.000000e-05, ulps = 5, mode = #stablehlo.result_accuracy_mode<TOLERANCE>>} : tensor<2xf32>
    return %0 : tensor<2xf32>
  }
})mlir";

  StablehloModuleBuilder mb;
  {
    Location funcLoc = fileLineColLoc(mb->getContext(), "main.mlir", 1, 1);
    func::FunctionBuilder fb(mb.get(), "main", funcLoc);
    auto type = makeTensorType(fb.getContext(), {2}, ElementType::F32);
    auto arg0 = func::Argument(fb, type);
    auto resultAccuracy =
        ResultAccuracyAttr::get(&fb.getContext(), /*atol=*/APFloat(1e-5),
                                /*rtol=*/APFloat(0.0), /*ulps=*/5);
    auto exp = Exp(arg0, resultAccuracy);
    func::Return(fb, {exp});
  }

  OwningOpRef<ModuleOp> module = mb->build();
  EXPECT_EQ(expected, debugString(*module));
}

TEST(MlirBuilderTest, FrontendAttributesAppend) {
  std::string expected = R"mlir(module {
  func.func @main(%arg0: tensor<2xf32>) -> tensor<2xf32> {
    %0 = stablehlo.exponential %arg0 {mhlo.frontend_attributes = {bar = "hello", foo = 123 : i32}} : tensor<2xf32>
    return %0 : tensor<2xf32>
  }
})mlir";

  StablehloModuleBuilder mb;
  {
    Location funcLoc = fileLineColLoc(mb->getContext(), "main.mlir", 1, 1);
    func::FunctionBuilder fb(mb.get(), "main", funcLoc);
    auto type = makeTensorType(fb.getContext(), {2}, ElementType::F32);
    auto arg0 = func::Argument(fb, type);
    auto exp = Exp(arg0);
    stablehlo::AttachFrontendAttribute(
        fb, exp, "foo", fb.getOpBuilder().getI32IntegerAttr(123));
    stablehlo::AttachFrontendAttribute(
        fb, exp, "bar", fb.getOpBuilder().getStringAttr("hello"));
    func::Return(fb, {exp});
  }

  OwningOpRef<ModuleOp> module = mb->build();
  EXPECT_EQ(expected, debugString(*module));
}

TEST(MlirBuilderTest, FrontendAttributesOverwrite) {
  std::string expected = R"mlir(module {
  func.func @main(%arg0: tensor<2xf32>) -> tensor<2xf32> {
    %0 = stablehlo.exponential %arg0 {mhlo.frontend_attributes = {foo = 456 : i32}} : tensor<2xf32>
    return %0 : tensor<2xf32>
  }
})mlir";

  StablehloModuleBuilder mb;
  {
    Location funcLoc = fileLineColLoc(mb->getContext(), "main.mlir", 1, 1);
    func::FunctionBuilder fb(mb.get(), "main", funcLoc);
    auto type = makeTensorType(fb.getContext(), {2}, ElementType::F32);
    auto arg0 = func::Argument(fb, type);
    auto exp = Exp(arg0);
    stablehlo::AttachFrontendAttribute(
        fb, exp, "foo", fb.getOpBuilder().getI32IntegerAttr(123));
    stablehlo::AttachFrontendAttribute(
        fb, exp, "foo", fb.getOpBuilder().getI32IntegerAttr(456));
    func::Return(fb, {exp});
  }

  OwningOpRef<ModuleOp> module = mb->build();
  EXPECT_EQ(expected, debugString(*module));
}

}  // namespace stablehlo
}  // namespace mlir
