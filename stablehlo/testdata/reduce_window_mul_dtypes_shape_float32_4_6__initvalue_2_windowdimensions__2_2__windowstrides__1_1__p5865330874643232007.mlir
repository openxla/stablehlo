// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<4x6xf32>
    %1 = call @expected() : () -> tensor<3x5xf32>
    %2 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %3 = "stablehlo.reduce_window"(%0, %2) ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %5 = stablehlo.multiply %arg0, %arg1 : tensor<f32>
      stablehlo.return %5 : tensor<f32>
    }) {window_dimensions = dense<2> : tensor<2xi64>} : (tensor<4x6xf32>, tensor<f32>) -> tensor<3x5xf32>
    %4 = stablehlo.custom_call @check.eq(%3, %1) : (tensor<3x5xf32>, tensor<3x5xf32>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> tensor<4x6xf32> {
    %0 = stablehlo.constant dense<[[3.46309638, 2.09357405, 0.159164399, 2.21717548, -5.09275532, 2.23330903], [3.31880307, 1.91353369, -0.626116395, -4.15517235, -1.83640027, 0.570109904], [-3.92634177, 4.49188948, -1.6027441, 1.64508104, -0.735530615, 2.85355926], [-0.435912549, 1.67701876, 3.77726054, -1.27700937, -1.78002596, 0.950089871]]> : tensor<4x6xf32>
    return %0 : tensor<4x6xf32>
  }
  func.func private @expected() -> tensor<3x5xf32> {
    %0 = stablehlo.constant dense<[[92.0874633, -0.798464179, 1.83620059, -172.321426, 23.8153839], [-224.008621, 17.2510071, -13.719099, -18.4660683, 4.39484501], [25.7860279, -91.2090911, 25.4362144, -5.5009594, 7.09918785]]> : tensor<3x5xf32>
    return %0 : tensor<3x5xf32>
  }
}

