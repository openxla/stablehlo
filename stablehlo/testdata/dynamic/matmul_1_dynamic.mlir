// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x8x4xf32> {mhlo.sharding = ""}, %arg2: tensor<4x5xf32> {mhlo.sharding = ""}) -> tensor<?x8x5xf32> {
    %0 = "stablehlo.dot_general"(%arg1, %arg2) {dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [0]>} : (tensor<?x8x4xf32>, tensor<4x5xf32>) -> tensor<?x8x5xf32>
    return %0 : tensor<?x8x5xf32>
  }
}

