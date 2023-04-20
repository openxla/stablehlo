// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x4x4xcomplex<f32>> {mhlo.sharding = ""}, %arg2: tensor<?x4x1xcomplex<f32>> {mhlo.sharding = ""}) -> tensor<?x4x1xcomplex<f32>> {
    %0 = "stablehlo.triangular_solve"(%arg1, %arg2) {left_side = true, lower = false, transpose_a = #stablehlo<transpose NO_TRANSPOSE>, unit_diagonal = true} : (tensor<?x4x4xcomplex<f32>>, tensor<?x4x1xcomplex<f32>>) -> tensor<?x4x1xcomplex<f32>>
    return %0 : tensor<?x4x1xcomplex<f32>>
  }
}

