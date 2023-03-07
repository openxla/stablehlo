// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x4x6xf32> {mhlo.sharding = ""}, %arg2: tensor<?x4x6xf32> {mhlo.sharding = ""}) -> tensor<?x4x6xf32> {
    %0 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2:2 = "stablehlo.reduce_window"(%arg2, %arg1, %0, %1) ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>, %arg5: tensor<f32>, %arg6: tensor<f32>):
      %3 = stablehlo.compare  LE, %arg3, %arg5,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %4 = stablehlo.select %3, %arg3, %arg5 : tensor<i1>, tensor<f32>
      %5 = stablehlo.select %3, %arg4, %arg6 : tensor<i1>, tensor<f32>
      stablehlo.return %4, %5 : tensor<f32>, tensor<f32>
    }) {padding = dense<[[0, 0], [0, 1], [0, 1]]> : tensor<3x2xi64>, window_dimensions = dense<[1, 2, 2]> : tensor<3xi64>} : (tensor<?x4x6xf32>, tensor<?x4x6xf32>, tensor<f32>, tensor<f32>) -> (tensor<?x4x6xf32>, tensor<?x4x6xf32>)
    return %2#1 : tensor<?x4x6xf32>
  }
}

