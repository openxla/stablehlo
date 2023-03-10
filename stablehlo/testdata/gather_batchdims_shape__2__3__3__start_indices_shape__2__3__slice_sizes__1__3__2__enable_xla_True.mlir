// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<2x3x3xf32>, tensor<2x3xi32>)
    %1 = call @expected() : () -> tensor<2x3x2xf32>
    %2 = "stablehlo.gather"(%0#0, %0#1) {dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], collapsed_slice_dims = [0], start_index_map = [0, 1, 2], index_vector_dim = 1>, slice_sizes = dense<[1, 3, 2]> : tensor<3xi64>} : (tensor<2x3x3xf32>, tensor<2x3xi32>) -> tensor<2x3x2xf32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<2x3x2xf32>, tensor<2x3x2xf32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<2x3x3xf32>, tensor<2x3xi32>) {
    %0 = stablehlo.constant dense<[[[-2.31606197, -1.45022011, -1.72503948], [-4.47900438, -5.43648243, -4.72877312], [4.36842155, -1.49977052, -2.34371066]], [[-2.62882113, -3.40511084, 0.60867834], [-2.19209099, -0.954817473, -0.967517852], [-0.497551709, 6.707040e-01, -6.8893342]]]> : tensor<2x3x3xf32>
    %1 = stablehlo.constant dense<[[0, 1, 0], [1, 2, 1]]> : tensor<2x3xi32>
    return %0, %1 : tensor<2x3x3xf32>, tensor<2x3xi32>
  }
  func.func private @expected() -> tensor<2x3x2xf32> {
    %0 = stablehlo.constant dense<[[[-2.31606197, -1.45022011], [-4.47900438, -5.43648243], [4.36842155, -1.49977052]], [[-3.40511084, 0.60867834], [-0.954817473, -0.967517852], [6.707040e-01, -6.8893342]]]> : tensor<2x3x2xf32>
    return %0 : tensor<2x3x2xf32>
  }
}

