// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<3xcomplex<f32>>, tensor<3x6xcomplex<f32>>)
    %1 = call @expected() : () -> tensor<6xcomplex<f32>>
    %2 = "stablehlo.dot_general"(%0#0, %0#1) {dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [0], rhs_contracting_dimensions = [0]>} : (tensor<3xcomplex<f32>>, tensor<3x6xcomplex<f32>>) -> tensor<6xcomplex<f32>>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<6xcomplex<f32>>, tensor<6xcomplex<f32>>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<3xcomplex<f32>>, tensor<3x6xcomplex<f32>>) {
    %0 = stablehlo.constant dense<[(-1.16007435,0.714724123), (-2.50378919,-1.98855698), (-4.02477503,1.20222187)]> : tensor<3xcomplex<f32>>
    %1 = stablehlo.constant dense<[[(-6.13875579,-1.96624207), (0.00381713198,2.145970e+00), (-1.84627652,1.14139175), (3.606990e+00,2.36349583), (1.30285442,-0.94945544), (-1.24096477,5.66305351)], [(-6.0828495,5.13352346), (1.46488667,0.216771096), (-4.11369085,-2.45818734), (-1.70778179,0.997769296), (-5.87785578,-3.00215626), (-4.75901222,2.81252575)], [(-4.69427109,-5.14227247), (1.35996532,0.979071915), (-3.29123688,6.74801731), (-1.86385632,0.893658101), (-5.41866875,5.21715975), (-0.857319951,0.613969504)]]> : tensor<3x6xcomplex<f32>>
    return %0, %1 : tensor<3xcomplex<f32>>, tensor<3x6xcomplex<f32>>
  }
  func.func private @expected() -> tensor<6xcomplex<f32>> {
    %0 = stablehlo.constant dense<[(59.0407486,12.1892357), (-11.4255266,-8.248080e+00), (11.8714809,-19.4246349), (6.81364917,-5.10355186), (23.450882,-6.27449798), (17.6129131,-8.53669261)]> : tensor<6xcomplex<f32>>
    return %0 : tensor<6xcomplex<f32>>
  }
}

