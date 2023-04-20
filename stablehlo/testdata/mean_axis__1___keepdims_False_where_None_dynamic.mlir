// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x8x4xf32> {mhlo.sharding = ""}) -> tensor<?x4xf32> {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%arg1 init: %0) across dimensions = [1] : (tensor<?x8x4xf32>, tensor<f32>) -> tensor<?x4xf32>
     reducer(%arg2: tensor<f32>, %arg3: tensor<f32>)  {
      %9 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %9 : tensor<f32>
    }
    %2 = stablehlo.constant dense<8.000000e+00> : tensor<f32>
    %3 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %4 = stablehlo.reshape %3 : (tensor<i32>) -> tensor<1xi32>
    %5 = stablehlo.constant dense<4> : tensor<1xi32>
    %6 = stablehlo.concatenate %4, %5, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %7 = stablehlo.dynamic_broadcast_in_dim %2, %6, dims = [] : (tensor<f32>, tensor<2xi32>) -> tensor<?x4xf32>
    %8 = stablehlo.divide %1, %7 : tensor<?x4xf32>
    return %8 : tensor<?x4xf32>
  }
}

