// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<2x3xi16>, tensor<i16>)
    %1 = call @expected() : () -> tensor<6x4xi16>
    %2 = stablehlo.pad %0#0, %0#1, low = [1, 0], high = [2, 1], interior = [1, 0] : (tensor<2x3xi16>, tensor<i16>) -> tensor<6x4xi16>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<6x4xi16>, tensor<6x4xi16>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<2x3xi16>, tensor<i16>) {
    %0 = stablehlo.constant dense<0> : tensor<2x3xi16>
    %1 = stablehlo.constant dense<0> : tensor<i16>
    return %0, %1 : tensor<2x3xi16>, tensor<i16>
  }
  func.func private @expected() -> tensor<6x4xi16> {
    %0 = stablehlo.constant dense<0> : tensor<6x4xi16>
    return %0 : tensor<6x4xi16>
  }
}
