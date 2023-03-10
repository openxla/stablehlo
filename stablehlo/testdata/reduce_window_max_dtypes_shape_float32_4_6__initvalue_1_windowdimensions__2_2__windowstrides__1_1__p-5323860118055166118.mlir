// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<4x6xf32>
    %1 = call @expected() : () -> tensor<3x5xf32>
    %2 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %3 = "stablehlo.reduce_window"(%0, %2) ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %5 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
      stablehlo.return %5 : tensor<f32>
    }) {window_dimensions = dense<2> : tensor<2xi64>} : (tensor<4x6xf32>, tensor<f32>) -> tensor<3x5xf32>
    %4 = stablehlo.custom_call @check.eq(%3, %1) : (tensor<3x5xf32>, tensor<3x5xf32>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> tensor<4x6xf32> {
    %0 = stablehlo.constant dense<[[-4.54097319, 3.79378533, 1.59863269, 4.24134684, 2.04231811, -4.34703112], [3.18803906, -1.92934394, -4.10865164, 0.685572564, 1.10662353, 3.13294077], [2.22856736, 4.13388252, 2.24903274, 1.48916948, 1.04856753, 3.84324169], [5.06065464, -1.385570e+00, -1.74880457, -1.28526604, -6.0323391, 2.10861874]]> : tensor<4x6xf32>
    return %0 : tensor<4x6xf32>
  }
  func.func private @expected() -> tensor<3x5xf32> {
    %0 = stablehlo.constant dense<[[3.79378533, 3.79378533, 4.24134684, 4.24134684, 3.13294077], [4.13388252, 4.13388252, 2.24903274, 1.48916948, 3.84324169], [5.06065464, 4.13388252, 2.24903274, 1.48916948, 3.84324169]]> : tensor<3x5xf32>
    return %0 : tensor<3x5xf32>
  }
}

