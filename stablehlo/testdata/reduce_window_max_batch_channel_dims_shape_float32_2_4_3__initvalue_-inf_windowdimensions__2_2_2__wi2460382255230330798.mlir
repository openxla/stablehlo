// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<2x4x3xf32>
    %1 = call @expected() : () -> tensor<1x3x2xf32>
    %2 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f32>) -> tensor<f32>
    %4 = "stablehlo.reduce_window"(%0, %3) ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %6 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
      stablehlo.return %6 : tensor<f32>
    }) {window_dimensions = dense<2> : tensor<3xi64>} : (tensor<2x4x3xf32>, tensor<f32>) -> tensor<1x3x2xf32>
    %5 = stablehlo.custom_call @check.eq(%4, %1) : (tensor<1x3x2xf32>, tensor<1x3x2xf32>) -> tensor<i1>
    return %5 : tensor<i1>
  }
  func.func private @inputs() -> tensor<2x4x3xf32> {
    %0 = stablehlo.constant dense<[[[-3.09693503, -3.45139885, 0.316036344], [2.54035425, -1.4493444, 1.34590113], [-1.37607467, 1.43668747, 0.610349417], [-1.372087, -0.964820623, -5.42558479]], [[-1.45048237, -3.48121524, 4.49751425], [4.92057419, 3.3822782, 0.991242051], [-1.03866565, -5.23930168, -0.414829791], [2.75383162, 2.04341269, -1.76491356]]]> : tensor<2x4x3xf32>
    return %0 : tensor<2x4x3xf32>
  }
  func.func private @expected() -> tensor<1x3x2xf32> {
    %0 = stablehlo.constant dense<[[[4.92057419, 4.49751425], [4.92057419, 3.3822782], [2.75383162, 2.04341269]]]> : tensor<1x3x2xf32>
    return %0 : tensor<1x3x2xf32>
  }
}

