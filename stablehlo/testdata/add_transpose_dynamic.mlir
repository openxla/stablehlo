// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x4xf32> {mhlo.sharding = ""}) -> tensor<?x4xf32> {
    %0 = stablehlo.cosine %arg1 : tensor<?x4xf32>
    %1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %2 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %3 = stablehlo.reshape %2 : (tensor<i32>) -> tensor<1xi32>
    %4 = stablehlo.constant dense<4> : tensor<1xi32>
    %5 = stablehlo.concatenate %3, %4, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %6 = stablehlo.dynamic_broadcast_in_dim %1, %5, dims = [] : (tensor<f32>, tensor<2xi32>) -> tensor<?x4xf32>
    %7 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.reduce(%6 init: %7) across dimensions = [0] : (tensor<?x4xf32>, tensor<f32>) -> tensor<4xf32>
     reducer(%arg2: tensor<f32>, %arg3: tensor<f32>)  {
      %19 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %19 : tensor<f32>
    }
    %9 = stablehlo.reshape %8 : (tensor<4xf32>) -> tensor<1x4xf32>
    %10 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = stablehlo.reduce(%9 init: %10) across dimensions = [0] : (tensor<1x4xf32>, tensor<f32>) -> tensor<4xf32>
     reducer(%arg2: tensor<f32>, %arg3: tensor<f32>)  {
      %19 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %19 : tensor<f32>
    }
    %12 = stablehlo.multiply %6, %0 : tensor<?x4xf32>
    %13 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %14 = stablehlo.reshape %13 : (tensor<i32>) -> tensor<1xi32>
    %15 = stablehlo.constant dense<4> : tensor<1xi32>
    %16 = stablehlo.concatenate %14, %15, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %17 = stablehlo.dynamic_broadcast_in_dim %11, %16, dims = [1] : (tensor<4xf32>, tensor<2xi32>) -> tensor<?x4xf32>
    %18 = stablehlo.add %12, %17 : tensor<?x4xf32>
    return %18 : tensor<?x4xf32>
  }
}

