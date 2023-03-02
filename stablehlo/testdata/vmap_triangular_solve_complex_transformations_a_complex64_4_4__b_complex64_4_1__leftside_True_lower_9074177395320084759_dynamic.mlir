// RUN: diff <(stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-opt %s --stablehlo-legalize-to-vhlo --vhlo-to-version=target=current -emit-bytecode | stablehlo-opt --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x4x4xcomplex<f32>> {mhlo.sharding = ""}, %arg2: tensor<?x4x1xcomplex<f32>> {mhlo.sharding = ""}) -> tensor<?x4x1xcomplex<f32>> {
    %0 = stablehlo.real %arg1 : (tensor<?x4x4xcomplex<f32>>) -> tensor<?x4x4xf32>
    %1 = stablehlo.imag %arg1 : (tensor<?x4x4xcomplex<f32>>) -> tensor<?x4x4xf32>
    %2 = stablehlo.negate %1 : tensor<?x4x4xf32>
    %3 = stablehlo.complex %0, %2 : tensor<?x4x4xcomplex<f32>>
    %4 = "stablehlo.triangular_solve"(%3, %arg2) {left_side = true, lower = false, transpose_a = #stablehlo<transpose NO_TRANSPOSE>, unit_diagonal = false} : (tensor<?x4x4xcomplex<f32>>, tensor<?x4x1xcomplex<f32>>) -> tensor<?x4x1xcomplex<f32>>
    return %4 : tensor<?x4x1xcomplex<f32>>
  }
}

