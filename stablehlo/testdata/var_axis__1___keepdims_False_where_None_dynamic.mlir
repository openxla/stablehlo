// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x8x4xf32> {mhlo.sharding = ""}) -> tensor<?x4xf32> {
    %0 = stablehlo.constant dense<0> : tensor<i64>
    %1 = call @_var(%arg0, %arg1, %0) : (tensor<i64>, tensor<?x8x4xf32>, tensor<i64>) -> tensor<?x4xf32>
    return %1 : tensor<?x4xf32>
  }
  func.func private @_var(%arg0: tensor<i64>, %arg1: tensor<?x8x4xf32>, %arg2: tensor<i64>) -> tensor<?x4xf32> {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%arg1 init: %0) across dimensions = [1] : (tensor<?x8x4xf32>, tensor<f32>) -> tensor<?x4xf32>
     reducer(%arg3: tensor<f32>, %arg4: tensor<f32>)  {
      %35 = stablehlo.add %arg3, %arg4 : tensor<f32>
      stablehlo.return %35 : tensor<f32>
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
    %16 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %17 = stablehlo.reshape %16 : (tensor<i32>) -> tensor<1xi32>
    %18 = stablehlo.constant dense<8> : tensor<1xi32>
    %19 = stablehlo.constant dense<4> : tensor<1xi32>
    %20 = stablehlo.concatenate %17, %18, %19, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %21 = stablehlo.dynamic_broadcast_in_dim %15, %20, dims = [0, 1, 2] : (tensor<?x1x4xf32>, tensor<3xi32>) -> tensor<?x8x4xf32>
    %22 = stablehlo.subtract %arg1, %21 : tensor<?x8x4xf32>
    %23 = stablehlo.multiply %22, %22 : tensor<?x8x4xf32>
    %24 = stablehlo.constant dense<8> : tensor<i64>
    %25 = stablehlo.subtract %24, %arg2 : tensor<i64>
    %26 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %27 = stablehlo.reduce(%23 init: %26) across dimensions = [1] : (tensor<?x8x4xf32>, tensor<f32>) -> tensor<?x4xf32>
     reducer(%arg3: tensor<f32>, %arg4: tensor<f32>)  {
      %35 = stablehlo.add %arg3, %arg4 : tensor<f32>
      stablehlo.return %35 : tensor<f32>
    }
    %28 = stablehlo.convert %25 : (tensor<i64>) -> tensor<f32>
    %29 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %30 = stablehlo.reshape %29 : (tensor<i32>) -> tensor<1xi32>
    %31 = stablehlo.constant dense<4> : tensor<1xi32>
    %32 = stablehlo.concatenate %30, %31, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %33 = stablehlo.dynamic_broadcast_in_dim %28, %32, dims = [] : (tensor<f32>, tensor<2xi32>) -> tensor<?x4xf32>
    %34 = stablehlo.divide %27, %33 : tensor<?x4xf32>
    return %34 : tensor<?x4xf32>
  }
}

