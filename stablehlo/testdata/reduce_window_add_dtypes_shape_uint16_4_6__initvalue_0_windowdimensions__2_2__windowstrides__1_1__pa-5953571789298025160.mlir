// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<4x6xui16>
    %1 = call @expected() : () -> tensor<3x5xui16>
    %2 = stablehlo.constant dense<0> : tensor<ui16>
    %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<ui16>) -> tensor<ui16>
    %4 = "stablehlo.reduce_window"(%0, %3) ({
    ^bb0(%arg0: tensor<ui16>, %arg1: tensor<ui16>):
      %6 = stablehlo.add %arg0, %arg1 : tensor<ui16>
      stablehlo.return %6 : tensor<ui16>
    }) {window_dimensions = dense<2> : tensor<2xi64>} : (tensor<4x6xui16>, tensor<ui16>) -> tensor<3x5xui16>
    %5 = stablehlo.custom_call @check.eq(%4, %1) : (tensor<3x5xui16>, tensor<3x5xui16>) -> tensor<i1>
    return %5 : tensor<i1>
  }
  func.func private @inputs() -> tensor<4x6xui16> {
    %0 = stablehlo.constant dense<[[4, 2, 0, 2, 1, 4], [4, 7, 5, 2, 0, 4], [0, 4, 4, 1, 2, 2], [1, 2, 5, 2, 1, 3]]> : tensor<4x6xui16>
    return %0 : tensor<4x6xui16>
  }
  func.func private @expected() -> tensor<3x5xui16> {
    %0 = stablehlo.constant dense<[[17, 14, 9, 5, 9], [15, 20, 12, 5, 8], [7, 15, 12, 6, 8]]> : tensor<3x5xui16>
    return %0 : tensor<3x5xui16>
  }
}

