// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x8x4x3x3x4xf32> {mhlo.sharding = ""}, %arg2: tensor<?x4x8x3x4x2xf32> {mhlo.sharding = ""}) -> tensor<?x8x4x3x2xf32> {
    %0 = "stablehlo.dot_general"(%arg1, %arg2) {dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [0, 1, 2], rhs_batching_dimensions = [0, 2, 1], lhs_contracting_dimensions = [5, 4], rhs_contracting_dimensions = [4, 3]>} : (tensor<?x8x4x3x3x4xf32>, tensor<?x4x8x3x4x2xf32>) -> tensor<?x8x4x3x2xf32>
    return %0 : tensor<?x8x4x3x2xf32>
  }
}

