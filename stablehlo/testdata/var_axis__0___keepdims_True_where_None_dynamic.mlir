// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x8x4xf32> {mhlo.sharding = ""}) -> tensor<1x8x4xf32> {
    %0 = stablehlo.constant dense<0> : tensor<i64>
    %1 = call @_var(%arg0, %arg1, %0) : (tensor<i64>, tensor<?x8x4xf32>, tensor<i64>) -> tensor<1x8x4xf32>
    return %1 : tensor<1x8x4xf32>
  }
  func.func private @_var(%arg0: tensor<i64>, %arg1: tensor<?x8x4xf32>, %arg2: tensor<i64>) -> tensor<1x8x4xf32> {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%arg1 init: %0) across dimensions = [0] : (tensor<?x8x4xf32>, tensor<f32>) -> tensor<8x4xf32>
     reducer(%arg3: tensor<f32>, %arg4: tensor<f32>)  {
      %21 = stablehlo.add %arg3, %arg4 : tensor<f32>
      stablehlo.return %21 : tensor<f32>
    }
    %2 = stablehlo.broadcast_in_dim %1, dims = [1, 2] : (tensor<8x4xf32>) -> tensor<1x8x4xf32>
    %3 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<f32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<f32>) -> tensor<1x8x4xf32>
    %5 = stablehlo.divide %2, %4 : tensor<1x8x4xf32>
    %6 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %7 = stablehlo.reshape %6 : (tensor<i32>) -> tensor<1xi32>
    %8 = stablehlo.constant dense<8> : tensor<1xi32>
    %9 = stablehlo.constant dense<4> : tensor<1xi32>
    %10 = stablehlo.concatenate %7, %8, %9, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %11 = stablehlo.dynamic_broadcast_in_dim %5, %10, dims = [0, 1, 2] : (tensor<1x8x4xf32>, tensor<3xi32>) -> tensor<?x8x4xf32>
    %12 = stablehlo.subtract %arg1, %11 : tensor<?x8x4xf32>
    %13 = stablehlo.multiply %12, %12 : tensor<?x8x4xf32>
    %14 = stablehlo.subtract %arg0, %arg2 : tensor<i64>
    %15 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %16 = stablehlo.reduce(%13 init: %15) across dimensions = [0] : (tensor<?x8x4xf32>, tensor<f32>) -> tensor<8x4xf32>
     reducer(%arg3: tensor<f32>, %arg4: tensor<f32>)  {
      %21 = stablehlo.add %arg3, %arg4 : tensor<f32>
      stablehlo.return %21 : tensor<f32>
    }
    %17 = stablehlo.broadcast_in_dim %16, dims = [1, 2] : (tensor<8x4xf32>) -> tensor<1x8x4xf32>
    %18 = stablehlo.convert %14 : (tensor<i64>) -> tensor<f32>
    %19 = stablehlo.broadcast_in_dim %18, dims = [] : (tensor<f32>) -> tensor<1x8x4xf32>
    %20 = stablehlo.divide %17, %19 : tensor<1x8x4xf32>
    return %20 : tensor<1x8x4xf32>
  }
}

