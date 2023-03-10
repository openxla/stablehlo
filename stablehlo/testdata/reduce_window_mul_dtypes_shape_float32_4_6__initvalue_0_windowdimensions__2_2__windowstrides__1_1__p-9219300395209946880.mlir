// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<4x6xf32>
    %1 = call @expected() : () -> tensor<3x5xf32>
    %2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3 = "stablehlo.reduce_window"(%0, %2) ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %5 = stablehlo.multiply %arg0, %arg1 : tensor<f32>
      stablehlo.return %5 : tensor<f32>
    }) {window_dimensions = dense<2> : tensor<2xi64>} : (tensor<4x6xf32>, tensor<f32>) -> tensor<3x5xf32>
    %4 = stablehlo.custom_call @check.eq(%3, %1) : (tensor<3x5xf32>, tensor<3x5xf32>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> tensor<4x6xf32> {
    %0 = stablehlo.constant dense<[[4.39043951, -2.52543116, 0.124689803, 1.88919508, 1.46674097, -2.21504807], [-0.839728355, 2.08033895, 1.45145273, -5.8984375, 2.67627215, 2.49952412], [-1.44695759, -5.64082146, -1.93143368, -3.3416779, 0.272155076, 5.45000362], [0.935949385, 1.8907696, 2.51160383, -1.95614636, 2.43069816, -1.72123182]]> : tensor<4x6xf32>
    return %0 : tensor<4x6xf32>
  }
  func.func private @expected() -> tensor<3x5xf32> {
    %0 = stablehlo.constant dense<[[0.000000e+00, -0.000000e+00, -0.000000e+00, -0.000000e+00, -0.000000e+00], [-0.000000e+00, 0.000000e+00, -0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, -0.000000e+00, 0.000000e+00, -0.000000e+00]]> : tensor<3x5xf32>
    return %0 : tensor<3x5xf32>
  }
}

