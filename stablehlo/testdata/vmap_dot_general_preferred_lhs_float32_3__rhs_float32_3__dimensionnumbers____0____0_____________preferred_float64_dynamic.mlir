// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x3xf32> {mhlo.sharding = ""}, %arg2: tensor<?x3xf32> {mhlo.sharding = ""}) -> tensor<?xf64> {
    %0 = "stablehlo.dot_general"(%arg1, %arg2) {dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [1]>} : (tensor<?x3xf32>, tensor<?x3xf32>) -> tensor<?xf64>
    return %0 : tensor<?xf64>
  }
}

