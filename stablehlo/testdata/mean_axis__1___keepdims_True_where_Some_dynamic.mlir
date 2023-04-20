// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x8x4xf32> {mhlo.sharding = ""}, %arg2: tensor<?x8x4xi1> {mhlo.sharding = ""}) -> tensor<?x1x4xf32> {
    %0 = stablehlo.convert %arg2 : (tensor<?x8x4xi1>) -> tensor<?x8x4xi32>
    %1 = stablehlo.convert %0 : (tensor<?x8x4xi32>) -> tensor<?x8x4xi64>
    %2 = stablehlo.constant dense<0> : tensor<i64>
    %3 = stablehlo.reduce(%1 init: %2) across dimensions = [1] : (tensor<?x8x4xi64>, tensor<i64>) -> tensor<?x4xi64>
     reducer(%arg3: tensor<i64>, %arg4: tensor<i64>)  {
      %22 = stablehlo.add %arg3, %arg4 : tensor<i64>
      stablehlo.return %22 : tensor<i64>
    }
    %4 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %5 = stablehlo.reshape %4 : (tensor<i32>) -> tensor<1xi32>
    %6 = stablehlo.constant dense<1> : tensor<1xi32>
    %7 = stablehlo.constant dense<4> : tensor<1xi32>
    %8 = stablehlo.concatenate %5, %6, %7, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %9 = stablehlo.dynamic_broadcast_in_dim %3, %8, dims = [0, 2] : (tensor<?x4xi64>, tensor<3xi32>) -> tensor<?x1x4xi64>
    %10 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = call @_where(%arg0, %arg2, %arg1, %10) : (tensor<i64>, tensor<?x8x4xi1>, tensor<?x8x4xf32>, tensor<f32>) -> tensor<?x8x4xf32>
    %12 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %13 = stablehlo.reduce(%11 init: %12) across dimensions = [1] : (tensor<?x8x4xf32>, tensor<f32>) -> tensor<?x4xf32>
     reducer(%arg3: tensor<f32>, %arg4: tensor<f32>)  {
      %22 = stablehlo.add %arg3, %arg4 : tensor<f32>
      stablehlo.return %22 : tensor<f32>
    }
    %14 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %15 = stablehlo.reshape %14 : (tensor<i32>) -> tensor<1xi32>
    %16 = stablehlo.constant dense<1> : tensor<1xi32>
    %17 = stablehlo.constant dense<4> : tensor<1xi32>
    %18 = stablehlo.concatenate %15, %16, %17, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %19 = stablehlo.dynamic_broadcast_in_dim %13, %18, dims = [0, 2] : (tensor<?x4xf32>, tensor<3xi32>) -> tensor<?x1x4xf32>
    %20 = stablehlo.convert %9 : (tensor<?x1x4xi64>) -> tensor<?x1x4xf32>
    %21 = stablehlo.divide %19, %20 : tensor<?x1x4xf32>
    return %21 : tensor<?x1x4xf32>
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

