// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x2xf32> {mhlo.sharding = ""}, %arg2: tensor<2x3xf32> {mhlo.sharding = ""}, %arg3: tensor<3x4xf32> {mhlo.sharding = ""}) -> tensor<?x4xf32> {
    %0 = call @_einsum(%arg0, %arg1, %arg2, %arg3) : (tensor<i64>, tensor<?x2xf32>, tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<?x4xf32>
    return %0 : tensor<?x4xf32>
  }
  func.func private @_einsum(%arg0: tensor<i64>, %arg1: tensor<?x2xf32>, %arg2: tensor<2x3xf32>, %arg3: tensor<3x4xf32>) -> tensor<?x4xf32> {
    %0 = "stablehlo.dot_general"(%arg3, %arg2) {dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [0], rhs_contracting_dimensions = [1]>} : (tensor<3x4xf32>, tensor<2x3xf32>) -> tensor<4x2xf32>
    %1 = "stablehlo.dot_general"(%arg1, %0) {dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [1]>} : (tensor<?x2xf32>, tensor<4x2xf32>) -> tensor<?x4xf32>
    return %1 : tensor<?x4xf32>
  }
}

