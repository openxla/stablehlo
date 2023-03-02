// RUN: diff <(stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-opt %s --stablehlo-legalize-to-vhlo --vhlo-to-version=target=current -emit-bytecode | stablehlo-opt --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x4x4xcomplex<f32>> {mhlo.sharding = ""}, %arg2: tensor<?x4x1xcomplex<f32>> {mhlo.sharding = ""}) -> tensor<?x4x1xcomplex<f32>> {
    %0 = "stablehlo.triangular_solve"(%arg1, %arg2) {left_side = true, lower = true, transpose_a = #stablehlo<transpose NO_TRANSPOSE>, unit_diagonal = false} : (tensor<?x4x4xcomplex<f32>>, tensor<?x4x1xcomplex<f32>>) -> tensor<?x4x1xcomplex<f32>>
    return %0 : tensor<?x4x1xcomplex<f32>>
  }
}

