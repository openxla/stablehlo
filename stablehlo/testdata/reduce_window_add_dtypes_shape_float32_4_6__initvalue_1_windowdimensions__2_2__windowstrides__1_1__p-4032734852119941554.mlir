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
      %5 = stablehlo.add %arg0, %arg1 : tensor<f32>
      stablehlo.return %5 : tensor<f32>
    }) {window_dimensions = dense<2> : tensor<2xi64>} : (tensor<4x6xf32>, tensor<f32>) -> tensor<3x5xf32>
    %4 = stablehlo.custom_call @check.eq(%3, %1) : (tensor<3x5xf32>, tensor<3x5xf32>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> tensor<4x6xf32> {
    %0 = stablehlo.constant dense<[[-1.40100789, -4.37216043, -1.56686234, 0.868020236, -3.37629104, 1.0409044], [0.965046346, -1.90812123, -6.60379791, 5.02317333, -0.422616273, -1.90688837], [0.352010161, -3.19368887, 4.38464117, 2.38747144, -4.60010481, 2.453340e+00], [-1.38386226, 3.87740374, 0.809796273, -2.3112185, -3.38834834, 2.87746906]]> : tensor<4x6xf32>
    return %0 : tensor<4x6xf32>
  }
  func.func private @expected() -> tensor<3x5xf32> {
    %0 = stablehlo.constant dense<[[-5.71624327, -13.450942, -1.27946663, 3.09228635, -3.66489124], [-2.78475356, -6.3209672, 6.19148827, 3.38792372, -3.47626925], [0.651862621, 6.87815237, 6.27069091, -6.91219997, -1.65764427]]> : tensor<3x5xf32>
    return %0 : tensor<3x5xf32>
  }
}

