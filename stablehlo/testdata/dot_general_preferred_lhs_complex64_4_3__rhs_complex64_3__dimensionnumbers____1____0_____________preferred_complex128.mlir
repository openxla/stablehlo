// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<4x3xcomplex<f32>>, tensor<3xcomplex<f32>>)
    %1 = call @expected() : () -> tensor<4xcomplex<f32>>
    %2 = "stablehlo.dot_general"(%0#0, %0#1) {dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<4x3xcomplex<f32>>, tensor<3xcomplex<f32>>) -> tensor<4xcomplex<f32>>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<4xcomplex<f32>>, tensor<4xcomplex<f32>>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x3xcomplex<f32>>, tensor<3xcomplex<f32>>) {
    %0 = stablehlo.constant dense<[[(5.78633165,7.04203319), (-2.48239541,4.14060736), (-4.227390e+00,-0.865227103)], [(1.3521533,3.03694582), (1.60163248,-2.4179635), (-1.69885457,-1.87064886)], [(-0.557790279,1.19244611), (-2.81902218,2.29712176), (2.20077109,2.2956183)], [(-2.13202953,1.99405694), (1.6221118,7.5080924), (-1.91944265,-4.50890923)]]> : tensor<4x3xcomplex<f32>>
    %1 = stablehlo.constant dense<[(5.06853104,-1.74682879), (0.86917305,-0.739572703), (4.8467226,1.15608156)]> : tensor<3xcomplex<f32>>
    return %0, %1 : tensor<4x3xcomplex<f32>>, tensor<3xcomplex<f32>>
  }
  func.func private @expected() -> tensor<4xcomplex<f32>> {
    %0 = stablehlo.constant dense<[(23.0453644,21.939127), (5.69103765,-1.28581047), (6.517097,24.7702789), (-4.45064497,-4.91507149)]> : tensor<4xcomplex<f32>>
    return %0 : tensor<4xcomplex<f32>>
  }
}

