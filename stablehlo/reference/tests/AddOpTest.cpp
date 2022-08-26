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

#include "gtest/gtest.h"
#include "stablehlo/reference/tests/TestUtils.h"

namespace mlir {
namespace stablehlo {

TEST(AddOpInterpreterTest, F16) {
  constexpr llvm::StringRef kModule = R"(
  module {
    func.func @main(%arg0: tensor<2x5xf16>, %arg1: tensor<2x5xf16>) -> tensor<2x5xf16> {
      %result = stablehlo.add %arg0, %arg1 : tensor<2x5xf16>
      func.return %result : tensor<2x5xf16>
    }
  })";
  runTestCase(kModule, {/*operands*/
                        {"0.0", "-0.0", "1.0", "0.125", "0.1", "3.141", "inf",
                         "inf", "-inf", "inf"},
                        {"0.0", "-0.0", "7.0", "0.75", "0.3", "3.141", "0",
                         "inf", "-inf", "-inf"},
                        /*expected result*/
                        {"0.0", "-0.0", "8.0", "0.875", "0.4", "6.282", "inf",
                         "inf", "-inf", "nan"}});
}

TEST(AddOpInterpreterTest, BF16) {
  constexpr llvm::StringRef kModule = R"(
  module {
    func.func @main(%arg0: tensor<2x5xbf16>, %arg1: tensor<2x5xbf16>) -> tensor<2x5xbf16> {
      %result = stablehlo.add %arg0, %arg1 : tensor<2x5xbf16>
      func.return %result : tensor<2x5xbf16>
    }
  })";
  runTestCase(kModule, {/*operands*/
                        {"0.0", "-0.0", "1.0", "0.125", "0.1", "3.140625",
                         "inf", "inf", "-inf", "inf"},
                        {"0.0", "-0.0", "7.0", "0.75", "0.3", "3.140625", "0",
                         "inf", "-inf", "-inf"},
                        /*expected result*/
                        {"0.0", "-0.0", "8.0", "0.875", "0.4", "6.28125", "inf",
                         "inf", "-inf", "nan"}});
}

TEST(AddOpInterpreterTest, F32) {
  constexpr llvm::StringRef kModule = R"(
  module {
    func.func @main(%arg0: tensor<2x5xf32>, %arg1: tensor<2x5xf32>) -> tensor<2x5xf32> {
      %result = stablehlo.add %arg0, %arg1 : tensor<2x5xf32>
      func.return %result : tensor<2x5xf32>
    }
  })";
  runTestCase(kModule, {/*operands*/
                        {"0.0", "-0.0", "1.0", "0.125", "0.1", "3.14159265",
                         "inf", "inf", "-inf", "inf"},
                        {"0.0", "-0.0", "7.0", "0.75", "0.3", "3.14159265", "0",
                         "inf", "-inf", "-inf"},
                        /*expected result*/
                        {"0.0", "-0.0", "8.0", "0.875", "0.4", "6.2831855",
                         "inf", "inf", "-inf", "nan"}});
}

TEST(AddOpInterpreterTest, F64) {
  constexpr llvm::StringRef kModule = R"(
  module {
    func.func @main(%arg0: tensor<2x5xf64>, %arg1: tensor<2x5xf64>) -> tensor<2x5xf64> {
      %result = stablehlo.add %arg0, %arg1 : tensor<2x5xf64>
      func.return %result : tensor<2x5xf64>
    }
  })";

  runTestCase(kModule, {/*operands*/
                        {"0.0", "-0.0", "1.0", "0.125", "0.1",
                         "3.14159265358979323846", "inf", "inf", "-inf", "inf"},
                        {"0.0", "-0.0", "7.0", "0.75", "0.3",
                         "3.14159265358979323846", "0", "inf", "-inf", "-inf"},
                        /*expected result*/
                        {"0.0", "-0.0", "8.0", "0.875", "0.4",
                         "6.283185307179586", "inf", "inf", "-inf", "nan"}});
}

TEST(AddOpInterpreterTest, SInt4) {
  constexpr llvm::StringRef kModule = R"(
  module {
    func.func @main(%arg0: tensor<5xi4>, %arg1: tensor<5xi4>) -> tensor<5xi4> {
      %result = stablehlo.add %arg0, %arg1 : tensor<5xi4>
      func.return %result : tensor<5xi4>
    }
  })";
  runTestCase(kModule, {/*operands*/
                        {"0", "1", "2", "-3", "0"},
                        {"-8", "-1", "2", "-3", "7"},
                        /*expected result*/
                        {"-8", "0", "4", "-6", "7"}});
}

TEST(AddOpInterpreterTest, UInt4) {
  constexpr llvm::StringRef kModule = R"(
  module {
    func.func @main(%arg0: tensor<2xui4>, %arg1: tensor<2xui4>) -> tensor<2xui4> {
      %result = stablehlo.add %arg0, %arg1 : tensor<2xui4>
      func.return %result : tensor<2xui4>
    }
  })";
  runTestCase(kModule, {/*operands*/
                        {"0", "2"},
                        {"15", "3"},
                        /*expected result*/
                        {"15", "5"}});
}

TEST(AddOpInterpreterTest, SInt8) {
  constexpr llvm::StringRef kModule = R"(
  module {
    func.func @main(%arg0: tensor<5xi8>, %arg1: tensor<5xi8>) -> tensor<5xi8> {
      %result = stablehlo.add %arg0, %arg1 : tensor<5xi8>
      func.return %result : tensor<5xi8>
    }
  })";
  runTestCase(kModule, {/*operands*/
                        {"0", "1", "8", "-9", "0"},
                        {"-128", "-1", "8", "-9", "127"},
                        /*expected result*/
                        {"-128", "0", "16", "-18", "127"}});
}

TEST(AddOpInterpreterTest, UInt8) {
  constexpr llvm::StringRef kModule = R"(
  module {
    func.func @main(%arg0: tensor<2xui8>, %arg1: tensor<2xui8>) -> tensor<2xui8> {
      %result = stablehlo.add %arg0, %arg1 : tensor<2xui8>
      func.return %result : tensor<2xui8>
    }
  })";
  runTestCase(kModule, {/*operands*/
                        {"0", "16"},
                        {"255", "16"},
                        /*expected result*/
                        {"255", "32"}});
}

TEST(AddOpInterpreterTest, SInt16) {
  constexpr llvm::StringRef kModule = R"(
  module {
    func.func @main(%arg0: tensor<5xi16>, %arg1: tensor<5xi16>) -> tensor<5xi16> {
      %result = stablehlo.add %arg0, %arg1 : tensor<5xi16>
      func.return %result : tensor<5xi16>
    }
  })";
  runTestCase(kModule, {/*operands*/
                        {"0", "1", "128", "-129", "0"},
                        {"-32768", "-1", "128", "-129", "32767"},
                        /*expected result*/
                        {"-32768", "0", "256", "-258", "32767"}});
}

TEST(AddOpInterpreterTest, UInt16) {
  constexpr llvm::StringRef kModule = R"(
  module {
    func.func @main(%arg0: tensor<2xui16>, %arg1: tensor<2xui16>) -> tensor<2xui16> {
      %result = stablehlo.add %arg0, %arg1 : tensor<2xui16>
      func.return %result : tensor<2xui16>
    }
  })";
  runTestCase(kModule, {/*operands*/
                        {"0", "256"},
                        {"65535", "256"},
                        /*expected result*/
                        {"65535", "512"}});
}

TEST(AddOpInterpreterTest, SInt32) {
  constexpr llvm::StringRef kModule = R"(
  module {
    func.func @main(%arg0: tensor<5xi32>, %arg1: tensor<5xi32>) -> tensor<5xi32> {
      %result = stablehlo.add %arg0, %arg1 : tensor<5xi32>
      func.return %result : tensor<5xi32>
    }
  })";
  runTestCase(kModule, {/*operands*/
                        {"0", "1", "32768", "-32769", "0"},
                        {"-2147483648", "-1", "32768", "-32769", "2147483647"},
                        /*expected result*/
                        {"-2147483648", "0", "65536", "-65538", "2147483647"}});
}

TEST(AddOpInterpreterTest, UInt32) {
  constexpr llvm::StringRef kModule = R"(
  module {
    func.func @main(%arg0: tensor<2xui32>, %arg1: tensor<2xui32>) -> tensor<2xui32> {
      %result = stablehlo.add %arg0, %arg1 : tensor<2xui32>
      func.return %result : tensor<2xui32>
    }
  })";
  runTestCase(kModule, {/*operands*/
                        {"0", "65536"},
                        {"4294967295", "65536"},
                        /*expected result*/
                        {"4294967295", "131072"}});
}

TEST(AddOpInterpreterTest, SInt64) {
  constexpr llvm::StringRef kModule = R"(
  module {
    func.func @main(%arg0: tensor<5xi64>, %arg1: tensor<5xi64>) -> tensor<5xi64> {
      %result = stablehlo.add %arg0, %arg1 : tensor<5xi64>
      func.return %result : tensor<5xi64>
    }
  })";
  runTestCase(kModule, {/*operands*/
                        {"0", "1", "2147483648", "-2147483649", "0"},
                        {"-9223372036854775808", "-1", "2147483648",
                         "-2147483649", "9223372036854775807"},
                        /*expected result*/
                        {"-9223372036854775808", "0", "4294967296",
                         "-4294967298", "9223372036854775807"}});
}

TEST(AddOpInterpreterTest, UInt64) {
  constexpr llvm::StringRef kModule = R"(
  module {
    func.func @main(%arg0: tensor<2xui64>, %arg1: tensor<2xui64>) -> tensor<2xui64> {
      %result = stablehlo.add %arg0, %arg1 : tensor<2xui64>
      func.return %result : tensor<2xui64>
    }
  })";
  runTestCase(kModule, {/*operands*/
                        {"0", "4294967296"},
                        {"18446744073709551615", "4294967296"},
                        /*expected result*/
                        {"18446744073709551615", "8589934592"}});
}

TEST(AddOpInterpreterTest, Complex32) {
  constexpr llvm::StringRef kModule = R"(
  module {
    func.func @main(%arg0: tensor<2xcomplex<f32>>, %arg1: tensor<2xcomplex<f32>>) -> tensor<2xcomplex<f32>> {
      %result = stablehlo.add %arg0, %arg1 : tensor<2xcomplex<f32>>
      func.return %result : tensor<2xcomplex<f32>>
    }
  })";
  runTestCase(kModule, {/*operands*/
                        {"1.5", "2.5", "7.5", "5.5"},
                        {"1.5", "2.5", "7.5", "5.5"},
                        /*expected result*/
                        {"3.0", "5.0", "15.0", "11.0"}});
}

TEST(AddOpInterpreterTest, Complex64) {
  constexpr llvm::StringRef kModule = R"(
  module {
    func.func @main(%arg0: tensor<2xcomplex<f64>>, %arg1: tensor<2xcomplex<f64>>) -> tensor<2xcomplex<f64>> {
      %result = stablehlo.add %arg0, %arg1 : tensor<2xcomplex<f64>>
      func.return %result : tensor<2xcomplex<f64>>
    }
  })";
  runTestCase(kModule, {/*operands*/
                        {"1.5", "2.5", "7.5", "5.5"},
                        {"1.5", "2.5", "7.5", "5.5"},
                        /*expected result*/
                        {"3.0", "5.0", "15.0", "11.0"}});
}

}  // namespace stablehlo
}  // namespace mlir
