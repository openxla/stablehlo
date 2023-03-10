// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<3x?xf32> {mhlo.sharding = ""}, %arg2: tensor<?x5xf32> {mhlo.sharding = ""}) -> tensor<3x5xf32> {
    %0 = call @_einsum(%arg0, %arg1, %arg2) : (tensor<i64>, tensor<3x?xf32>, tensor<?x5xf32>) -> tensor<3x5xf32>
    return %0 : tensor<3x5xf32>
  }
  func.func private @_einsum(%arg0: tensor<i64>, %arg1: tensor<3x?xf32>, %arg2: tensor<?x5xf32>) -> tensor<3x5xf32> {
    %0 = "stablehlo.dot_general"(%arg1, %arg2) {dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<3x?xf32>, tensor<?x5xf32>) -> tensor<3x5xf32>
    return %0 : tensor<3x5xf32>
  }
}

