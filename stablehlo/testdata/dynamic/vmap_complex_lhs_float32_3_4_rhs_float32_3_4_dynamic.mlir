// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x3x4xf32> {mhlo.sharding = ""}, %arg2: tensor<?x3x4xf32> {mhlo.sharding = ""}) -> tensor<?x3x4xcomplex<f32>> {
    %0 = stablehlo.complex %arg1, %arg2 : tensor<?x3x4xcomplex<f32>>
    return %0 : tensor<?x3x4xcomplex<f32>>
  }
}

