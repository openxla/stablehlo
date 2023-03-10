// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<2x4x3xf32>, tensor<2x1x3xf32>)
    %1 = call @expected() : () -> tensor<2x4x3xf32>
    %2 = stablehlo.broadcast_in_dim %0#1, dims = [0, 1, 2] : (tensor<2x1x3xf32>) -> tensor<2x4x3xf32>
    %3 = stablehlo.divide %0#0, %2 : tensor<2x4x3xf32>
    %4 = stablehlo.custom_call @check.eq(%3, %1) : (tensor<2x4x3xf32>, tensor<2x4x3xf32>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<2x4x3xf32>, tensor<2x1x3xf32>) {
    %0 = stablehlo.constant dense<[[[-2.66624355, -2.3699677, 2.11741066], [-8.13989543, 1.31505847, 1.49205363], [0.44794336, -3.37213826, -6.38927936], [-3.03567147, -0.791732251, -0.56633085]], [[-1.76294041, 3.25217557, -6.35648203], [1.5057143, 1.4200381, -3.02599382], [-1.47315633, -3.69056392, 1.13986015], [-1.0625931, 2.26376176, 0.833067059]]]> : tensor<2x4x3xf32>
    %1 = stablehlo.constant dense<[[[-1.07903361, 0.73738116, -0.378743678]], [[1.47470522, -0.96192944, -3.62243295]]]> : tensor<2x1x3xf32>
    return %0, %1 : tensor<2x4x3xf32>, tensor<2x1x3xf32>
  }
  func.func private @expected() -> tensor<2x4x3xf32> {
    %0 = stablehlo.constant dense<[[[2.47095513, -3.21403337, -5.5906167], [7.543690e+00, 1.78341746, -3.93948126], [-0.415133834, -4.57312775, 16.8696651], [2.81332421, -1.0737083, 1.4952879]], [[-1.19545269, -3.38088775, 1.7547549], [1.02102733, -1.47623932, 0.835348486], [-0.998949706, 3.83662653, -0.314667016], [-0.720546126, -2.35335541, -0.229974464]]]> : tensor<2x4x3xf32>
    return %0 : tensor<2x4x3xf32>
  }
}
