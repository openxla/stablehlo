// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x4x6xf32> {mhlo.sharding = ""}) -> tensor<?x3x5xf32> {
    %0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %1 = "stablehlo.reduce_window"(%arg1, %0) ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %2 = stablehlo.maximum %arg2, %arg3 : tensor<f32>
      stablehlo.return %2 : tensor<f32>
    }) {window_dimensions = array<i64: 1, 2, 2>} : (tensor<?x4x6xf32>, tensor<f32>) -> tensor<?x3x5xf32>
    return %1 : tensor<?x3x5xf32>
  }
}

