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
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "testing/base/public/gunit.h"
#include "llvm/ADT/DenseMap.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"

namespace mlir {

TEST(AttrTypeBuilderUtilTest, TestMakeTensorType) {
  MLIRContext context;
  llvm::DenseMap<std::pair<SmallVector<int64_t>, ElementType>, std::string>
      testCaseMap = {
          {{{}, ElementType::PRED}, "tensor<i1>"},
          {{{}, ElementType::I8}, "tensor<i8>"},
          {{{}, ElementType::I16}, "tensor<i16>"},
          {{{}, ElementType::I32}, "tensor<i32>"},
          {{{}, ElementType::I64}, "tensor<i64>"},
          {{{}, ElementType::UI8}, "tensor<ui8>"},
          {{{}, ElementType::UI16}, "tensor<ui16>"},
          {{{}, ElementType::UI32}, "tensor<ui32>"},
          {{{}, ElementType::UI64}, "tensor<ui64>"},
          {{{}, ElementType::BF16}, "tensor<bf16>"},
          {{{}, ElementType::F16}, "tensor<f16>"},
          {{{}, ElementType::F32}, "tensor<f32>"},
          {{{}, ElementType::F64}, "tensor<f64>"},
          {{{}, ElementType::F4E2M1FN}, "tensor<f4E2M1FN>"},
          {{{}, ElementType::F6E2M3FN}, "tensor<f6E2M3FN>"},
          {{{}, ElementType::F6E3M2FN}, "tensor<f6E3M2FN>"},
          {{{}, ElementType::F8E3M4}, "tensor<f8E3M4>"},
          {{{}, ElementType::F8E4M3}, "tensor<f8E4M3>"},
          {{{}, ElementType::F8E4M3FN}, "tensor<f8E4M3FN>"},
          {{{}, ElementType::F8E4M3FNUZ}, "tensor<f8E4M3FNUZ>"},
          {{{}, ElementType::F8E4M3B11FNUZ}, "tensor<f8E4M3B11FNUZ>"},
          {{{}, ElementType::F8E5M2}, "tensor<f8E5M2>"},
          {{{}, ElementType::F8E5M2FNUZ}, "tensor<f8E5M2FNUZ>"},
          {{{}, ElementType::F8E8M0FNU}, "tensor<f8E8M0FNU>"},
          {{{1}, ElementType::F64}, "tensor<1xf64>"},
          {{{1, 2, 3}, ElementType::F64}, "tensor<1x2x3xf64>"},
      };
  for (auto& [inputs, value] : testCaseMap) {
    RankedTensorType type =
        makeTensorType(context, inputs.first, inputs.second);
    Type mlir_element_type = getElementType(context, inputs.second);
    RankedTensorType type2 =
        makeTensorType(context, inputs.first, mlir_element_type);
    EXPECT_EQ(type, type2);
    EXPECT_EQ(value, debugString(type));
  }
}

TEST(AttrTypeBuilderUtilTest, TestMakeConstantSplat_Integer) {
  MLIRContext context;

  // Init with Int
  auto i32_type = makeTensorType(context, {}, ElementType::I32);
  EXPECT_EQ(mlir::debugString(makeConstant(1, i32_type)),
            "dense<1> : tensor<i32>");

  // Init with APSInt
  EXPECT_EQ(mlir::debugString(makeConstant(APSInt::get(1), i32_type)),
            "dense<1> : tensor<i32>");

  // Init with Float
  EXPECT_EQ(mlir::debugString(makeConstant(1.0, i32_type)),
            "dense<1> : tensor<i32>");
}

TEST(AttrTypeBuilderUtilTest, TestMakeConstantSplat_Float) {
  MLIRContext context;

  // Float
  auto f32_type = makeTensorType(context, {}, ElementType::F32);
  EXPECT_EQ(mlir::debugString(makeConstant(1.0, f32_type)),
            "dense<1.000000e+00> : tensor<f32>");

  // Init with APFloat
  FloatType f32_type2 = cast<FloatType>(f32_type.getElementType());
  auto inf = APFloat::getInf(f32_type2.getFloatSemantics());
  EXPECT_EQ(mlir::debugString(makeConstant(inf, f32_type)),
            "dense<0x7F800000> : tensor<f32>");

  // Init with Int
  EXPECT_EQ(mlir::debugString(makeConstant(1, f32_type)),
            "dense<1.000000e+00> : tensor<f32>");
}

TEST(AttrTypeBuilderUtilTest, TestMakeConstantSplat_Complex) {
  MLIRContext context;

  // Complex
  auto c32_type = makeTensorType(context, {}, ElementType::COMPLEXF32);
  EXPECT_EQ(
      mlir::debugString(makeConstant(std::complex<float>(1.0, 2.0), c32_type)),
      "dense<(1.000000e+00,2.000000e+00)> : tensor<complex<f32>>");

  auto c64_type = makeTensorType(context, {}, ElementType::COMPLEXF64);
  EXPECT_EQ(
      mlir::debugString(makeConstant(std::complex<double>(1.0, 2.0), c64_type)),
      "dense<(1.000000e+00,2.000000e+00)> : tensor<complex<f64>>");

  // Init with int & float
  EXPECT_EQ(mlir::debugString(makeConstant(1, c32_type)),
            "dense<(1.000000e+00,0.000000e+00)> : tensor<complex<f32>>");
  EXPECT_EQ(mlir::debugString(makeConstant(1.0, c32_type)),
            "dense<(1.000000e+00,0.000000e+00)> : tensor<complex<f32>>");
}

TEST(AttrTypeBuilderUtilTest, TestMakeConstantSplatIntLimits) {
  MLIRContext context;

  // Test 64 bitwidth values and i8 values, everything in between should be
  // uninteresting.

  // Int8
  auto i8_type = makeTensorType(context, {}, ElementType::I8);
  EXPECT_EQ(mlir::debugString(
                makeConstant(std::numeric_limits<int8_t>::max(), i8_type)),
            "dense<127> : tensor<i8>");
  EXPECT_EQ(mlir::debugString(
                makeConstant(std::numeric_limits<int8_t>::min(), i8_type)),
            "dense<-128> : tensor<i8>");

  // uint8
  auto u8_type = makeTensorType(context, {}, ElementType::UI8);
  EXPECT_EQ(mlir::debugString(
                makeConstant(std::numeric_limits<uint8_t>::max(), u8_type)),
            "dense<255> : tensor<ui8>");
  EXPECT_EQ(mlir::debugString(
                makeConstant(std::numeric_limits<uint8_t>::min(), u8_type)),
            "dense<0> : tensor<ui8>");

  // int64
  auto i64_type = makeTensorType(context, {}, ElementType::I64);
  EXPECT_EQ(mlir::debugString(
                makeConstant(std::numeric_limits<int64_t>::max(), i64_type)),
            "dense<9223372036854775807> : tensor<i64>");
  EXPECT_EQ(mlir::debugString(
                makeConstant(std::numeric_limits<int64_t>::min(), i64_type)),
            "dense<-9223372036854775808> : tensor<i64>");

  // uint64
  auto u64_type = makeTensorType(context, {}, ElementType::UI64);
  EXPECT_EQ(mlir::debugString(
                makeConstant(std::numeric_limits<uint64_t>::max(), u64_type)),
            "dense<18446744073709551615> : tensor<ui64>");
  EXPECT_EQ(mlir::debugString(
                makeConstant(std::numeric_limits<uint64_t>::min(), u64_type)),
            "dense<0> : tensor<ui64>");
}

TEST(AttrTypeBuilderUtilTest, TestMakeConstantSplatFloatLimits) {
  MLIRContext context;

  // Test 64 bitwidth values and f32 values, everything in between should be
  // uninteresting, i.e. should behave like f32.

  // Float16
  auto f32_type = makeTensorType(context, {}, ElementType::F32);
  EXPECT_EQ(mlir::debugString(
                makeConstant(std::numeric_limits<float>::max(), f32_type)),
            "dense<3.40282347E+38> : tensor<f32>");
  EXPECT_EQ(mlir::debugString(
                makeConstant(std::numeric_limits<float>::min(), f32_type)),
            "dense<1.17549435E-38> : tensor<f32>");

  // Float64
  auto f64_type = makeTensorType(context, {}, ElementType::F64);
  EXPECT_EQ(mlir::debugString(
                makeConstant(std::numeric_limits<double>::max(), f64_type)),
            "dense<1.7976931348623157E+308> : tensor<f64>");
  EXPECT_EQ(mlir::debugString(
                makeConstant(std::numeric_limits<double>::min(), f64_type)),
            "dense<2.2250738585072014E-308> : tensor<f64>");
}

TEST(AttrTypeBuilderUtilTest, TestMakeConstantArray_Int) {
  MLIRContext context;

  // Int -- Vector
  auto i32_type = makeTensorType(context, {2}, ElementType::I32);
  EXPECT_EQ(
      mlir::debugString(makeConstant(std::vector<int32_t>{1, 2}, i32_type)),
      "dense<[1, 2]> : tensor<2xi32>");

  // Init with float
  EXPECT_EQ(
      mlir::debugString(makeConstant(ArrayRef<float>{1.0, 2.0}, i32_type)),
      "dense<[1, 2]> : tensor<2xi32>");

  // Init with APSInt
  EXPECT_EQ(mlir::debugString(makeConstant(
                ArrayRef<APSInt>{APSInt::get(1), APSInt::get(2)}, i32_type)),
            "dense<[1, 2]> : tensor<2xi32>");
}

TEST(AttrTypeBuilderUtilTest, TestMakeConstantArray_Float) {
  MLIRContext context;

  // Float - ArrayRef
  auto f32_type = makeTensorType(context, {2}, ElementType::F32);
  EXPECT_EQ(
      mlir::debugString(makeConstant(ArrayRef<float>{1.0, 2.0}, f32_type)),
      "dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf32>");

  // Init with APFloat
  FloatType f32_type2 = cast<FloatType>(f32_type.getElementType());
  auto inf = APFloat::getInf(f32_type2.getFloatSemantics());
  auto nan = APFloat::getNaN(f32_type2.getFloatSemantics());
  EXPECT_EQ(
      mlir::debugString(makeConstant(ArrayRef<APFloat>{inf, nan}, f32_type)),
      "dense<[0x7F800000, 0x7FC00000]> : tensor<2xf32>");

  // Init with Int
  EXPECT_EQ(mlir::debugString(makeConstant(ArrayRef<int32_t>{1, 2}, f32_type)),
            "dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf32>");
}

TEST(AttrTypeBuilderUtilTest, TestMakeConstantArray_Complex) {
  MLIRContext context;

  // Complex
  auto c32_type = makeTensorType(context, {2}, ElementType::COMPLEXF32);
  std::vector<std::complex<float>> complexValues = {{1.0, 2.0}, {3.0, 4.0}};
  EXPECT_EQ(mlir::debugString(makeConstant(complexValues, c32_type)),
            "dense<[(1.000000e+00,2.000000e+00), (3.000000e+00,4.000000e+00)]> "
            ": tensor<2xcomplex<f32>>");

  auto c64_type = makeTensorType(context, {2}, ElementType::COMPLEXF64);
  EXPECT_EQ(mlir::debugString(makeConstant(complexValues, c64_type)),
            "dense<[(1.000000e+00,2.000000e+00), (3.000000e+00,4.000000e+00)]> "
            ": tensor<2xcomplex<f64>>");

  // Init with int & float
  EXPECT_EQ(mlir::debugString(makeConstant(ArrayRef<int32_t>{1, 2}, c32_type)),
            "dense<[(1.000000e+00,0.000000e+00), (2.000000e+00,0.000000e+00)]> "
            ": tensor<2xcomplex<f32>>");
  EXPECT_EQ(
      mlir::debugString(makeConstant(ArrayRef<double>{1.0, 2.0}, c32_type)),
      "dense<[(1.000000e+00,0.000000e+00), (2.000000e+00,0.000000e+00)]> : "
      "tensor<2xcomplex<f32>>");
}

TEST(AttrTypeBuilderUtilTest, TestMakeEmptyLiteral) {
  MLIRContext context;

  // Float - ArrayRef
  auto f32_type = makeTensorType(context, {0}, ElementType::F32);
  EXPECT_EQ(mlir::debugString(makeConstant(ArrayRef<double>{}, f32_type)),
            "dense<> : tensor<0xf32>");

  // Int -- Vector
  auto i32_type = makeTensorType(context, {1, 0}, ElementType::I32);
  EXPECT_EQ(mlir::debugString(makeConstant(std::vector<int32_t>{}, i32_type)),
            "dense<> : tensor<1x0xi32>");
}
}  // namespace mlir
