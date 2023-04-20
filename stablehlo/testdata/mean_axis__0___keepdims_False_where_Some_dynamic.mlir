// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x8x4xf32> {mhlo.sharding = ""}, %arg2: tensor<?x8x4xi1> {mhlo.sharding = ""}) -> tensor<8x4xf32> {
    %0 = stablehlo.convert %arg2 : (tensor<?x8x4xi1>) -> tensor<?x8x4xi32>
    %1 = stablehlo.convert %0 : (tensor<?x8x4xi32>) -> tensor<?x8x4xi64>
    %2 = stablehlo.constant dense<0> : tensor<i64>
    %3 = stablehlo.reduce(%1 init: %2) across dimensions = [0] : (tensor<?x8x4xi64>, tensor<i64>) -> tensor<8x4xi64>
     reducer(%arg3: tensor<i64>, %arg4: tensor<i64>)  {
      %10 = stablehlo.add %arg3, %arg4 : tensor<i64>
      stablehlo.return %10 : tensor<i64>
    }
    %4 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %5 = call @_where(%arg0, %arg2, %arg1, %4) : (tensor<i64>, tensor<?x8x4xi1>, tensor<?x8x4xf32>, tensor<f32>) -> tensor<?x8x4xf32>
    %6 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %7 = stablehlo.reduce(%5 init: %6) across dimensions = [0] : (tensor<?x8x4xf32>, tensor<f32>) -> tensor<8x4xf32>
     reducer(%arg3: tensor<f32>, %arg4: tensor<f32>)  {
      %10 = stablehlo.add %arg3, %arg4 : tensor<f32>
      stablehlo.return %10 : tensor<f32>
    }
    %8 = stablehlo.convert %3 : (tensor<8x4xi64>) -> tensor<8x4xf32>
    %9 = stablehlo.divide %7, %8 : tensor<8x4xf32>
    return %9 : tensor<8x4xf32>
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

