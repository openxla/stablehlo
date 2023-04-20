// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<3x10xf32>, tensor<3x2xi32>)
    %1 = call @expected() : () -> tensor<3x5xf32>
    %2 = "stablehlo.gather"(%0#0, %0#1) {dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0, 1], index_vector_dim = 1>, slice_sizes = dense<[1, 5]> : tensor<2xi64>} : (tensor<3x10xf32>, tensor<3x2xi32>) -> tensor<3x5xf32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<3x5xf32>, tensor<3x5xf32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<3x10xf32>, tensor<3x2xi32>) {
    %0 = stablehlo.constant dense<[[-0.786750376, -0.429459691, -2.42140698, 0.0205181241, -0.394114822, -2.58621716, -1.07088399, 3.29197717, -3.44814229, -0.25225088], [1.27824605, -2.20641971, 1.13592541, 2.04215646, -1.61209357, 3.22753859, -1.28165495, 3.17407966, 2.02299929, 2.47564316], [0.905838906, 3.71254492, 1.97064459, 3.77753663, 1.49392521, 4.79311323, 3.70975041, -1.04468286, 3.31870532, 1.45112896]]> : tensor<3x10xf32>
    %1 = stablehlo.constant dense<[[0, 0], [1, 8], [2, 0]]> : tensor<3x2xi32>
    return %0, %1 : tensor<3x10xf32>, tensor<3x2xi32>
  }
  func.func private @expected() -> tensor<3x5xf32> {
    %0 = stablehlo.constant dense<[[-0.786750376, -0.429459691, -2.42140698, 0.0205181241, -0.394114822], [3.22753859, -1.28165495, 3.17407966, 2.02299929, 2.47564316], [0.905838906, 3.71254492, 1.97064459, 3.77753663, 1.49392521]]> : tensor<3x5xf32>
    return %0 : tensor<3x5xf32>
  }
}

