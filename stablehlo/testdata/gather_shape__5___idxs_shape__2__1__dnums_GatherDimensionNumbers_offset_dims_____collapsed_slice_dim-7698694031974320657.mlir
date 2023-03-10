// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<5xf32>, tensor<2x1xi32>)
    %1 = call @expected() : () -> tensor<2xf32>
    %2 = "stablehlo.gather"(%0#0, %0#1) {dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = dense<1> : tensor<1xi64>} : (tensor<5xf32>, tensor<2x1xi32>) -> tensor<2xf32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<2xf32>, tensor<2xf32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<5xf32>, tensor<2x1xi32>) {
    %0 = stablehlo.constant dense<[-4.45134878, -1.72604203, 5.85744715, -1.34194362, 0.103943698]> : tensor<5xf32>
    %1 = stablehlo.constant dense<[[0], [2]]> : tensor<2x1xi32>
    return %0, %1 : tensor<5xf32>, tensor<2x1xi32>
  }
  func.func private @expected() -> tensor<2xf32> {
    %0 = stablehlo.constant dense<[-4.45134878, 5.85744715]> : tensor<2xf32>
    return %0 : tensor<2xf32>
  }
}

