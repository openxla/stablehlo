// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x8x4xf32> {mhlo.sharding = ""}) -> tensor<1x1x1xf32> {
    %0 = stablehlo.constant dense<0> : tensor<i64>
    %1 = call @_var(%arg0, %arg1, %0) : (tensor<i64>, tensor<?x8x4xf32>, tensor<i64>) -> tensor<1x1x1xf32>
    return %1 : tensor<1x1x1xf32>
  }
  func.func private @_var(%arg0: tensor<i64>, %arg1: tensor<?x8x4xf32>, %arg2: tensor<i64>) -> tensor<1x1x1xf32> {
    %0 = stablehlo.constant dense<32> : tensor<i64>
    %1 = stablehlo.multiply %arg0, %0 : tensor<i64>
    %2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3 = stablehlo.reduce(%arg1 init: %2) across dimensions = [0, 1, 2] : (tensor<?x8x4xf32>, tensor<f32>) -> tensor<f32>
     reducer(%arg3: tensor<f32>, %arg4: tensor<f32>)  {
      %25 = stablehlo.add %arg3, %arg4 : tensor<f32>
      stablehlo.return %25 : tensor<f32>
    }
    %4 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<f32>) -> tensor<1x1x1xf32>
    %5 = stablehlo.convert %1 : (tensor<i64>) -> tensor<f32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [] : (tensor<f32>) -> tensor<1x1x1xf32>
    %7 = stablehlo.divide %4, %6 : tensor<1x1x1xf32>
    %8 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %9 = stablehlo.reshape %8 : (tensor<i32>) -> tensor<1xi32>
    %10 = stablehlo.constant dense<8> : tensor<1xi32>
    %11 = stablehlo.constant dense<4> : tensor<1xi32>
    %12 = stablehlo.concatenate %9, %10, %11, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %13 = stablehlo.dynamic_broadcast_in_dim %7, %12, dims = [0, 1, 2] : (tensor<1x1x1xf32>, tensor<3xi32>) -> tensor<?x8x4xf32>
    %14 = stablehlo.subtract %arg1, %13 : tensor<?x8x4xf32>
    %15 = stablehlo.multiply %14, %14 : tensor<?x8x4xf32>
    %16 = stablehlo.constant dense<32> : tensor<i64>
    %17 = stablehlo.multiply %arg0, %16 : tensor<i64>
    %18 = stablehlo.subtract %17, %arg2 : tensor<i64>
    %19 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %20 = stablehlo.reduce(%15 init: %19) across dimensions = [0, 1, 2] : (tensor<?x8x4xf32>, tensor<f32>) -> tensor<f32>
     reducer(%arg3: tensor<f32>, %arg4: tensor<f32>)  {
      %25 = stablehlo.add %arg3, %arg4 : tensor<f32>
      stablehlo.return %25 : tensor<f32>
    }
    %21 = stablehlo.broadcast_in_dim %20, dims = [] : (tensor<f32>) -> tensor<1x1x1xf32>
    %22 = stablehlo.convert %18 : (tensor<i64>) -> tensor<f32>
    %23 = stablehlo.broadcast_in_dim %22, dims = [] : (tensor<f32>) -> tensor<1x1x1xf32>
    %24 = stablehlo.divide %21, %23 : tensor<1x1x1xf32>
    return %24 : tensor<1x1x1xf32>
  }
}

