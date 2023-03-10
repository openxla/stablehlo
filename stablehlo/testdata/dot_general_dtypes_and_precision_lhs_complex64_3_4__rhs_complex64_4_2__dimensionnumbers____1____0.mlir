// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<3x4xcomplex<f32>>, tensor<4x2xcomplex<f32>>)
    %1 = call @expected() : () -> tensor<3x2xcomplex<f32>>
    %2 = "stablehlo.dot_general"(%0#0, %0#1) {dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<3x4xcomplex<f32>>, tensor<4x2xcomplex<f32>>) -> tensor<3x2xcomplex<f32>>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<3x2xcomplex<f32>>, tensor<3x2xcomplex<f32>>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<3x4xcomplex<f32>>, tensor<4x2xcomplex<f32>>) {
    %0 = stablehlo.constant dense<[[(2.96413302,-0.792680561), (1.94621408,-1.64191747), (2.166991,9.196680e-01), (0.342380762,3.30923128)], [(-2.5651238,-2.35528278), (-2.78961849,-0.360368907), (3.28392339,-4.68180227), (0.168843135,-6.008680e+00)], [(-0.610931098,4.27244806), (-1.0353235,0.220729128), (2.30340123,4.5588026), (5.83238363,0.389628828)]]> : tensor<3x4xcomplex<f32>>
    %1 = stablehlo.constant dense<[[(-4.96502829,-1.00413394), (-2.74682832,-1.12770426)], [(-0.869917035,-3.75533843), (0.301074713,1.94771695)], [(-4.67332363,-1.09390116), (4.2482748,-0.785265982)], [(-4.72472668,0.891028523), (-4.96662712,1.96558392)]]> : tensor<4x2xcomplex<f32>>
    return %0, %1 : tensor<3x4xcomplex<f32>>, tensor<4x2xcomplex<f32>>
  }
  func.func private @expected() -> tensor<3x2xcomplex<f32>> {
    %0 = stablehlo.constant dense<[[(-37.0592613,-26.9195881), (-3.528820e+00,-11.4263754)], [(-4.4677763,71.8863297), (25.4984379,11.5267868)], [(-24.6282921,-37.3719177), (-10.6132154,14.0903702)]]> : tensor<3x2xcomplex<f32>>
    return %0 : tensor<3x2xcomplex<f32>>
  }
}

