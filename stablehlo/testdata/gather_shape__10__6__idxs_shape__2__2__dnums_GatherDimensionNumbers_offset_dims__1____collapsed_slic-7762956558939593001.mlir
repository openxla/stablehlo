// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<10x6xf32>, tensor<2x2xi32>)
    %1 = call @expected() : () -> tensor<2x3xf32>
    %2 = "stablehlo.gather"(%0#0, %0#1) {dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0, 1], index_vector_dim = 1>, slice_sizes = dense<[1, 3]> : tensor<2xi64>} : (tensor<10x6xf32>, tensor<2x2xi32>) -> tensor<2x3xf32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<10x6xf32>, tensor<2x2xi32>) {
    %0 = stablehlo.constant dense<[[3.70376086, -2.32510591, -1.99206495, -0.776150584, -1.28069758, 1.53095305], [4.254500e+00, 2.01416397, -1.37674105, -2.95107651, 1.79171216, 3.70664334], [-4.70540047, 1.86079466, -0.0389865339, 0.519325137, 0.993043422, 2.45253563], [-1.64374018, -4.46355247, -2.61922121, -1.25144911, -0.758273184, -0.619571567], [0.760658919, -0.846748232, -0.224860594, -0.453169256, -2.77163863, -4.63377285], [-3.29197121, 3.36990237, -2.68048525, 0.746385097, 4.05497313, 1.62619972], [-1.2534517, -0.0589894205, -6.8161869, -0.606249153, -0.126775339, 0.788002848], [-2.84779382, 4.66777229, 0.357156366, 2.44225931, 3.86339879, 0.127061546], [-0.131096423, -2.3385644, -5.0672822, 1.84735632, -0.60150218, 1.54275787], [1.72991562, 5.786930e-01, 1.5892092, -2.5360136, -3.90111494, -0.914829075]]> : tensor<10x6xf32>
    %1 = stablehlo.constant dense<[[0, 2], [1, 0]]> : tensor<2x2xi32>
    return %0, %1 : tensor<10x6xf32>, tensor<2x2xi32>
  }
  func.func private @expected() -> tensor<2x3xf32> {
    %0 = stablehlo.constant dense<[[-1.99206495, -0.776150584, -1.28069758], [4.254500e+00, 2.01416397, -1.37674105]]> : tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
  }
}

