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
#include "reference/tests/TestUtils.h"

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
    func.func @main(%arg0: tensor<2x2x2xi4>, %arg1: tensor<2x2x2xi4>) -> tensor<2x2x2xi4> {
      %result = stablehlo.add %arg0, %arg1 : tensor<2x2x2xi4>
      func.return %result : tensor<2x2x2xi4>
    }
  })";
  runTestCase(kModule, {/*operands*/
                        {"0", "1", "2", "-3", "7", "-8", "7", "-8"},
                        {"-8", "-1", "2", "-3", "1", "-1", "7", "-8"},
                        /*expected result*/
                        {"-8", "0", "4", "-6", "-8", "7", "-2", "0"}});
}

TEST(AddOpInterpreterTest, UInt4) {
  constexpr llvm::StringRef kModule = R"(
  module {
    func.func @main(%arg0: tensor<2x2xui4>, %arg1: tensor<2x2xui4>) -> tensor<2x2xui4> {
      %result = stablehlo.add %arg0, %arg1 : tensor<2x2xui4>
      func.return %result : tensor<2x2xui4>
    }
  })";
  runTestCase(kModule, {/*operands*/
                        {"0", "7", "15", "15"},
                        {"15", "7", "1", "15"},
                        /*expected result*/
                        {"15", "14", "0", "14"}});
}

TEST(AddOpInterpreterTest, SInt8) {
  constexpr llvm::StringRef kModule = R"(
  module {
    func.func @main(%arg0: tensor<1x9x1xi8>, %arg1: tensor<1x9x1xi8>) -> tensor<1x9x1xi8> {
      %result = stablehlo.add %arg0, %arg1 : tensor<1x9x1xi8>
      func.return %result : tensor<1x9x1xi8>
    }
  })";
  runTestCase(kModule,
              {/*operands*/
               {"1", "2", "-128", "-128", "16", "-16", "16", "127", "-127"},
               {"127", "127", "-1", "-128", "16", "-16", "-16", "0", "0"},
               /*expected result*/
               {"-128", "-127", "127", "0", "32", "-32", "0", "127", "-127"}});
}

TEST(AddOpInterpreterTest, UInt8) {
  constexpr llvm::StringRef kModule = R"(
  module {
    func.func @main(%arg0: tensor<2x2xui8>, %arg1: tensor<2x2xui8>) -> tensor<2x2xui8> {
      %result = stablehlo.add %arg0, %arg1 : tensor<2x2xui8>
      func.return %result : tensor<2x2xui8>
    }
  })";
  runTestCase(kModule, {/*operands*/
                        {"1", "2", "255", "4"},
                        {"255", "255", "255", "10"},
                        /*expected result*/
                        {"0", "1", "254", "14"}});
}

TEST(AddOpInterpreterTest, SInt16) {
  constexpr llvm::StringRef kModule = R"(
  module {
    func.func @main(%arg0: tensor<1x9x1xi16>, %arg1: tensor<1x9x1xi16>) -> tensor<1x9x1xi16> {
      %result = stablehlo.add %arg0, %arg1 : tensor<1x9x1xi16>
      func.return %result : tensor<1x9x1xi16>
    }
  })";
  runTestCase(
      kModule,
      {/*operands*/
       {"1", "2", "-32768", "-32768", "16", "-16", "16", "32767", "-32768"},
       {"32767", "32767", "-1", "-32768", "16", "-16", "-16", "0", "0"},
       /*expected result*/
       {"-32768", "-32767", "32767", "0", "32", "-32", "0", "32767",
        "-32768"}});
}

TEST(AddOpInterpreterTest, UInt16) {
  constexpr llvm::StringRef kModule = R"(
  module {
    func.func @main(%arg0: tensor<2x2xui16>, %arg1: tensor<2x2xui16>) -> tensor<2x2xui16> {
      %result = stablehlo.add %arg0, %arg1 : tensor<2x2xui16>
      func.return %result : tensor<2x2xui16>
    }
  })";
  runTestCase(kModule, {/*operands*/
                        {"1", "2", "65535", "4"},
                        {"65535", "65535", "65535", "10"},
                        /*expected result*/
                        {"0", "1", "65534", "14"}});
}

TEST(AddOpInterpreterTest, SInt32) {
  constexpr llvm::StringRef kModule = R"(
  module {
    func.func @main(%arg0: tensor<1x9x1xi32>, %arg1: tensor<1x9x1xi32>) -> tensor<1x9x1xi32> {
      %result = stablehlo.add %arg0, %arg1 : tensor<1x9x1xi32>
      func.return %result : tensor<1x9x1xi32>
    }
  })";
  runTestCase(kModule, {/*operands*/
                        {"1", "2", "-2147483648", "-2147483648", "16", "-16",
                         "16", "2147483647", "-2147483648"},
                        {"2147483647", "2147483647", "-1", "-2147483648", "16",
                         "-16", "-16", "0", "0"},
                        /*expected result*/
                        {"-2147483648", "-2147483647", "2147483647", "0", "32",
                         "-32", "0", "2147483647", "-2147483648"}});
}

TEST(AddOpInterpreterTest, UInt32) {
  constexpr llvm::StringRef kModule = R"(
  module {
    func.func @main(%arg0: tensor<2x2xui32>, %arg1: tensor<2x2xui32>) -> tensor<2x2xui32> {
      %result = stablehlo.add %arg0, %arg1 : tensor<2x2xui32>
      func.return %result : tensor<2x2xui32>
    }
  })";
  runTestCase(kModule, {/*operands*/
                        {"1", "2", "4294967295", "4"},
                        {"4294967295", "4294967295", "4294967295", "10"},
                        /*expected result*/
                        {"0", "1", "4294967294", "14"}});
}

TEST(AddOpInterpreterTest, SInt64) {
  constexpr llvm::StringRef kModule = R"(
  module {
    func.func @main(%arg0: tensor<1x9x1xi64>, %arg1: tensor<1x9x1xi64>) -> tensor<1x9x1xi64> {
      %result = stablehlo.add %arg0, %arg1 : tensor<1x9x1xi64>
      func.return %result : tensor<1x9x1xi64>
    }
  })";
  runTestCase(
      kModule,
      {/*operands*/
       {"1", "2", "-9223372036854775808", "-9223372036854775808", "16", "-16",
        "16", "9223372036854775807", "-9223372036854775808"},
       {"9223372036854775807", "9223372036854775807", "-1",
        "-9223372036854775808", "16", "-16", "-16", "0", "0"},
       /*expected result*/
       {"-9223372036854775808", "-9223372036854775807", "9223372036854775807",
        "0", "32", "-32", "0", "9223372036854775807", "-9223372036854775808"}});
}

TEST(AddOpInterpreterTest, UInt64) {
  constexpr llvm::StringRef kModule = R"(
  module {
    func.func @main(%arg0: tensor<2x2xui64>, %arg1: tensor<2x2xui64>) -> tensor<2x2xui64> {
      %result = stablehlo.add %arg0, %arg1 : tensor<2x2xui64>
      func.return %result : tensor<2x2xui64>
    }
  })";
  runTestCase(kModule, {/*operands*/
                        {"1", "2", "18446744073709551615", "4"},
                        {"18446744073709551615", "18446744073709551615",
                         "18446744073709551615", "10"},
                        /*expected result*/
                        {"0", "1", "18446744073709551614", "14"}});
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
