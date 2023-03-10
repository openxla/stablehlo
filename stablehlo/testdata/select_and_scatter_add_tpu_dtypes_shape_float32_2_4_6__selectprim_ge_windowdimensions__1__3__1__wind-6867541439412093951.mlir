// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<2x1x6xf32>, tensor<2x4x6xf32>)
    %1 = call @expected() : () -> tensor<2x4x6xf32>
    %2 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %3 = stablehlo.pad %0#1, %2, low = [0, 0, 0], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<2x4x6xf32>, tensor<f32>) -> tensor<2x4x6xf32>
    %4 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %5 = "stablehlo.select_and_scatter"(%3, %0#0, %4) ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %8 = stablehlo.compare  GE, %arg0, %arg1,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      stablehlo.return %8 : tensor<i1>
    }, {
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %8 = stablehlo.add %arg0, %arg1 : tensor<f32>
      stablehlo.return %8 : tensor<f32>
    }) {window_dimensions = dense<[1, 3, 1]> : tensor<3xi64>, window_strides = dense<[1, 2, 1]> : tensor<3xi64>} : (tensor<2x4x6xf32>, tensor<2x1x6xf32>, tensor<f32>) -> tensor<2x4x6xf32>
    %6 = "stablehlo.slice"(%5) {limit_indices = dense<[2, 4, 6]> : tensor<3xi64>, start_indices = dense<0> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>} : (tensor<2x4x6xf32>) -> tensor<2x4x6xf32>
    %7 = stablehlo.custom_call @check.eq(%6, %1) : (tensor<2x4x6xf32>, tensor<2x4x6xf32>) -> tensor<i1>
    return %7 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<2x1x6xf32>, tensor<2x4x6xf32>) {
    %0 = stablehlo.constant dense<[[[5.00026178, 0.589292765, 1.39804709, 1.39804173, 4.93898249, -2.75036192]], [[-1.32685876, -6.3086977, 0.699323416, -0.0865199342, 3.94539046, 5.97099829]]]> : tensor<2x1x6xf32>
    %1 = stablehlo.constant dense<[[[0.4515073, 0.00846019387, 2.25597739, 0.706758857, -1.17247009, 1.04833162], [0.231415734, 2.128829, -2.23328233, 3.24836373, -0.220539719, -3.38092875], [1.64468348, -5.89511919, -3.56841087, 1.98182213, -5.91910648, -2.01114607], [-6.04198027, -2.97861886, -5.27701616, -1.19638765, 5.80710411, -0.545798123]], [[10.3336058, 0.689076364, -5.45823336, 2.75139356, -2.94259286, 1.52800179], [0.596507847, -4.17665577, -0.932225763, 0.735892474, -2.86531806, 1.00733972], [0.207674831, -0.302060485, -4.11529636, 1.67699647, 4.89687157, 8.15038681], [4.24055529, -5.131030e-01, -0.0718542337, -2.7699573, -1.55350304, 1.95404911]]]> : tensor<2x4x6xf32>
    return %0, %1 : tensor<2x1x6xf32>, tensor<2x4x6xf32>
  }
  func.func private @expected() -> tensor<2x4x6xf32> {
    %0 = stablehlo.constant dense<[[[0.000000e+00, 0.000000e+00, 1.39804709, 0.000000e+00, 0.000000e+00, -2.75036192], [0.000000e+00, 0.589292765, 0.000000e+00, 1.39804173, 4.93898249, 0.000000e+00], [5.00026178, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]], [[-1.32685876, -6.3086977, 0.000000e+00, -0.0865199342, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.699323416, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 3.94539046, 5.97099829], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]]]> : tensor<2x4x6xf32>
    return %0 : tensor<2x4x6xf32>
  }
}

