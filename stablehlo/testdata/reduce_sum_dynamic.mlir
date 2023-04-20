// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x5xf32> {mhlo.sharding = ""}) -> tensor<?x1xf32> {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%arg1 init: %0) across dimensions = [1] : (tensor<?x5xf32>, tensor<f32>) -> tensor<?xf32>
     reducer(%arg2: tensor<f32>, %arg3: tensor<f32>)  {
      %7 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %7 : tensor<f32>
    }
    %2 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %3 = stablehlo.reshape %2 : (tensor<i32>) -> tensor<1xi32>
    %4 = stablehlo.constant dense<1> : tensor<1xi32>
    %5 = stablehlo.concatenate %3, %4, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %6 = stablehlo.dynamic_broadcast_in_dim %1, %5, dims = [0] : (tensor<?xf32>, tensor<2xi32>) -> tensor<?x1xf32>
    return %6 : tensor<?x1xf32>
  }
}

