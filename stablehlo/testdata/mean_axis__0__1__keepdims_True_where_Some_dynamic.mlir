// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x8x4xf32> {mhlo.sharding = ""}, %arg2: tensor<?x8x4xi1> {mhlo.sharding = ""}) -> tensor<1x1x4xf32> {
    %0 = stablehlo.convert %arg2 : (tensor<?x8x4xi1>) -> tensor<?x8x4xi32>
    %1 = stablehlo.convert %0 : (tensor<?x8x4xi32>) -> tensor<?x8x4xi64>
    %2 = stablehlo.constant dense<0> : tensor<i64>
    %3 = stablehlo.reduce(%1 init: %2) across dimensions = [0, 1] : (tensor<?x8x4xi64>, tensor<i64>) -> tensor<4xi64>
     reducer(%arg3: tensor<i64>, %arg4: tensor<i64>)  {
      %12 = stablehlo.add %arg3, %arg4 : tensor<i64>
      stablehlo.return %12 : tensor<i64>
    }
    %4 = stablehlo.broadcast_in_dim %3, dims = [2] : (tensor<4xi64>) -> tensor<1x1x4xi64>
    %5 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = call @_where(%arg0, %arg2, %arg1, %5) : (tensor<i64>, tensor<?x8x4xi1>, tensor<?x8x4xf32>, tensor<f32>) -> tensor<?x8x4xf32>
    %7 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.reduce(%6 init: %7) across dimensions = [0, 1] : (tensor<?x8x4xf32>, tensor<f32>) -> tensor<4xf32>
     reducer(%arg3: tensor<f32>, %arg4: tensor<f32>)  {
      %12 = stablehlo.add %arg3, %arg4 : tensor<f32>
      stablehlo.return %12 : tensor<f32>
    }
    %9 = stablehlo.broadcast_in_dim %8, dims = [2] : (tensor<4xf32>) -> tensor<1x1x4xf32>
    %10 = stablehlo.convert %4 : (tensor<1x1x4xi64>) -> tensor<1x1x4xf32>
    %11 = stablehlo.divide %9, %10 : tensor<1x1x4xf32>
    return %11 : tensor<1x1x4xf32>
  }
  func.func private @_where(%arg0: tensor<i64>, %arg1: tensor<?x8x4xi1>, %arg2: tensor<?x8x4xf32>, %arg3: tensor<f32>) -> tensor<?x8x4xf32> {
    %0 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %1 = stablehlo.reshape %0 : (tensor<i32>) -> tensor<1xi32>
    %2 = stablehlo.constant dense<8> : tensor<1xi32>
    %3 = stablehlo.constant dense<4> : tensor<1xi32>
    %4 = stablehlo.concatenate %1, %2, %3, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %5 = stablehlo.dynamic_broadcast_in_dim %arg3, %4, dims = [] : (tensor<f32>, tensor<3xi32>) -> tensor<?x8x4xf32>
    %6 = stablehlo.select %arg1, %arg2, %5 : tensor<?x8x4xi1>, tensor<?x8x4xf32>
    return %6 : tensor<?x8x4xf32>
  }
}

