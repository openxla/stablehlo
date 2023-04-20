// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x8x4xf32> {mhlo.sharding = ""}) -> tensor<?x1x4xf32> {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%arg1 init: %0) across dimensions = [1] : (tensor<?x8x4xf32>, tensor<f32>) -> tensor<?x4xf32>
     reducer(%arg2: tensor<f32>, %arg3: tensor<f32>)  {
      %16 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %16 : tensor<f32>
    }
    %2 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %3 = stablehlo.reshape %2 : (tensor<i32>) -> tensor<1xi32>
    %4 = stablehlo.constant dense<1> : tensor<1xi32>
    %5 = stablehlo.constant dense<4> : tensor<1xi32>
    %6 = stablehlo.concatenate %3, %4, %5, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %7 = stablehlo.dynamic_broadcast_in_dim %1, %6, dims = [0, 2] : (tensor<?x4xf32>, tensor<3xi32>) -> tensor<?x1x4xf32>
    %8 = stablehlo.constant dense<8.000000e+00> : tensor<f32>
    %9 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %10 = stablehlo.reshape %9 : (tensor<i32>) -> tensor<1xi32>
    %11 = stablehlo.constant dense<1> : tensor<1xi32>
    %12 = stablehlo.constant dense<4> : tensor<1xi32>
    %13 = stablehlo.concatenate %10, %11, %12, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %14 = stablehlo.dynamic_broadcast_in_dim %8, %13, dims = [] : (tensor<f32>, tensor<3xi32>) -> tensor<?x1x4xf32>
    %15 = stablehlo.divide %7, %14 : tensor<?x1x4xf32>
    return %15 : tensor<?x1x4xf32>
  }
}

