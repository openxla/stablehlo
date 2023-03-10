// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<10x5xf32>, tensor<3x1xi32>)
    %1 = call @expected() : () -> tensor<3x3xf32>
    %2 = "stablehlo.gather"(%0#0, %0#1) {dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = dense<[1, 3]> : tensor<2xi64>} : (tensor<10x5xf32>, tensor<3x1xi32>) -> tensor<3x3xf32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<10x5xf32>, tensor<3x1xi32>) {
    %0 = stablehlo.constant dense<[[3.27917194, 4.79832602, -0.10540507, 3.0128758, -0.239777446], [2.15583777, 1.20524621, -5.21279478, 1.90036821, -5.13092661], [3.04004383, -0.748676181, 2.70175195, 1.52100611, -3.00484538], [3.73827457, 0.19627282, 1.94314909, 1.35509837, -3.43813014], [-3.82101965, -1.52074528, 3.47338939, -1.92120409, 0.261425197], [3.99755049, 1.19325948, -4.27102518, 0.404079616, 2.05883861], [-0.426693857, 3.07045269, 3.41785836, -3.19426727, -4.46876669], [2.52909493, 5.09040689, -0.238279715, -0.449262351, 2.13594961], [-3.88575697, -3.08637547, 3.126508, 1.62036669, -3.84586406], [-1.86297441, -6.78266811, -0.103852905, 2.65204668, 5.41461182]]> : tensor<10x5xf32>
    %1 = stablehlo.constant dense<[[0], [2], [1]]> : tensor<3x1xi32>
    return %0, %1 : tensor<10x5xf32>, tensor<3x1xi32>
  }
  func.func private @expected() -> tensor<3x3xf32> {
    %0 = stablehlo.constant dense<[[3.27917194, 4.79832602, -0.10540507], [3.04004383, -0.748676181, 2.70175195], [2.15583777, 1.20524621, -5.21279478]]> : tensor<3x3xf32>
    return %0 : tensor<3x3xf32>
  }
}

