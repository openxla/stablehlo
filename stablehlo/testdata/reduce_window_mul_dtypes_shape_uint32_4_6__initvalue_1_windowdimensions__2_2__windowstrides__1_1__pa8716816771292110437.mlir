// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<4x6xui32>
    %1 = call @expected() : () -> tensor<3x5xui32>
    %2 = stablehlo.constant dense<1> : tensor<ui32>
    %3 = "stablehlo.reduce_window"(%0, %2) ({
    ^bb0(%arg0: tensor<ui32>, %arg1: tensor<ui32>):
      %5 = stablehlo.multiply %arg0, %arg1 : tensor<ui32>
      stablehlo.return %5 : tensor<ui32>
    }) {window_dimensions = dense<2> : tensor<2xi64>} : (tensor<4x6xui32>, tensor<ui32>) -> tensor<3x5xui32>
    %4 = stablehlo.custom_call @check.eq(%3, %1) : (tensor<3x5xui32>, tensor<3x5xui32>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> tensor<4x6xui32> {
    %0 = stablehlo.constant dense<[[1, 2, 1, 1, 4, 6], [0, 2, 2, 0, 8, 2], [5, 0, 6, 1, 3, 1], [1, 0, 4, 0, 1, 0]]> : tensor<4x6xui32>
    return %0 : tensor<4x6xui32>
  }
  func.func private @expected() -> tensor<3x5xui32> {
    %0 = stablehlo.constant dense<[[0, 8, 0, 0, 384], [0, 0, 0, 0, 48], [0, 0, 0, 0, 0]]> : tensor<3x5xui32>
    return %0 : tensor<3x5xui32>
  }
}

