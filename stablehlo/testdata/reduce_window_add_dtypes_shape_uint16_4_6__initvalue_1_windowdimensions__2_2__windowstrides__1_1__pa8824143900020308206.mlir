// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<4x6xui16>
    %1 = call @expected() : () -> tensor<3x5xui16>
    %2 = stablehlo.constant dense<1> : tensor<ui16>
    %3 = "stablehlo.reduce_window"(%0, %2) ({
    ^bb0(%arg0: tensor<ui16>, %arg1: tensor<ui16>):
      %5 = stablehlo.add %arg0, %arg1 : tensor<ui16>
      stablehlo.return %5 : tensor<ui16>
    }) {window_dimensions = array<i64: 2, 2>} : (tensor<4x6xui16>, tensor<ui16>) -> tensor<3x5xui16>
    %4 = stablehlo.custom_call @check.eq(%3, %1) : (tensor<3x5xui16>, tensor<3x5xui16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> tensor<4x6xui16> {
    %0 = stablehlo.constant dense<[[0, 1, 1, 0, 3, 5], [0, 4, 1, 0, 4, 0], [1, 3, 3, 4, 4, 1], [3, 4, 2, 1, 2, 1]]> : tensor<4x6xui16>
    return %0 : tensor<4x6xui16>
  }
  func.func private @expected() -> tensor<3x5xui16> {
    %0 = stablehlo.constant dense<[[6, 8, 3, 8, 13], [9, 12, 9, 13, 10], [12, 13, 11, 12, 9]]> : tensor<3x5xui16>
    return %0 : tensor<3x5xui16>
  }
}

