// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x8x4xf32> {mhlo.sharding = ""}, %arg2: tensor<?x8x4xi1> {mhlo.sharding = ""}) -> tensor<1x8x4xf32> {
    %0 = stablehlo.constant dense<0> : tensor<i64>
    %1 = call @_var(%arg0, %arg1, %0, %arg2) : (tensor<i64>, tensor<?x8x4xf32>, tensor<i64>, tensor<?x8x4xi1>) -> tensor<1x8x4xf32>
    return %1 : tensor<1x8x4xf32>
  }
  func.func private @_var(%arg0: tensor<i64>, %arg1: tensor<?x8x4xf32>, %arg2: tensor<i64>, %arg3: tensor<?x8x4xi1>) -> tensor<1x8x4xf32> {
    %0 = stablehlo.convert %arg3 : (tensor<?x8x4xi1>) -> tensor<?x8x4xi32>
    %1 = stablehlo.convert %0 : (tensor<?x8x4xi32>) -> tensor<?x8x4xf32>
    %2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3 = stablehlo.reduce(%1 init: %2) across dimensions = [0] : (tensor<?x8x4xf32>, tensor<f32>) -> tensor<8x4xf32>
     reducer(%arg4: tensor<f32>, %arg5: tensor<f32>)  {
      %33 = stablehlo.add %arg4, %arg5 : tensor<f32>
      stablehlo.return %33 : tensor<f32>
    }
    %4 = stablehlo.broadcast_in_dim %3, dims = [1, 2] : (tensor<8x4xf32>) -> tensor<1x8x4xf32>
    %5 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = call @_where(%arg0, %arg3, %arg1, %5) : (tensor<i64>, tensor<?x8x4xi1>, tensor<?x8x4xf32>, tensor<f32>) -> tensor<?x8x4xf32>
    %7 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.reduce(%6 init: %7) across dimensions = [0] : (tensor<?x8x4xf32>, tensor<f32>) -> tensor<8x4xf32>
     reducer(%arg4: tensor<f32>, %arg5: tensor<f32>)  {
      %33 = stablehlo.add %arg4, %arg5 : tensor<f32>
      stablehlo.return %33 : tensor<f32>
    }
    %9 = stablehlo.broadcast_in_dim %8, dims = [1, 2] : (tensor<8x4xf32>) -> tensor<1x8x4xf32>
    %10 = stablehlo.divide %9, %4 : tensor<1x8x4xf32>
    %11 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %12 = stablehlo.reshape %11 : (tensor<i32>) -> tensor<1xi32>
    %13 = stablehlo.constant dense<8> : tensor<1xi32>
    %14 = stablehlo.constant dense<4> : tensor<1xi32>
    %15 = stablehlo.concatenate %12, %13, %14, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %16 = stablehlo.dynamic_broadcast_in_dim %10, %15, dims = [0, 1, 2] : (tensor<1x8x4xf32>, tensor<3xi32>) -> tensor<?x8x4xf32>
    %17 = stablehlo.subtract %arg1, %16 : tensor<?x8x4xf32>
    %18 = stablehlo.multiply %17, %17 : tensor<?x8x4xf32>
    %19 = stablehlo.convert %arg3 : (tensor<?x8x4xi1>) -> tensor<?x8x4xi32>
    %20 = stablehlo.convert %19 : (tensor<?x8x4xi32>) -> tensor<?x8x4xf32>
    %21 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %22 = stablehlo.reduce(%20 init: %21) across dimensions = [0] : (tensor<?x8x4xf32>, tensor<f32>) -> tensor<8x4xf32>
     reducer(%arg4: tensor<f32>, %arg5: tensor<f32>)  {
      %33 = stablehlo.add %arg4, %arg5 : tensor<f32>
      stablehlo.return %33 : tensor<f32>
    }
    %23 = stablehlo.broadcast_in_dim %22, dims = [1, 2] : (tensor<8x4xf32>) -> tensor<1x8x4xf32>
    %24 = stablehlo.convert %arg2 : (tensor<i64>) -> tensor<f32>
    %25 = stablehlo.broadcast_in_dim %24, dims = [] : (tensor<f32>) -> tensor<1x8x4xf32>
    %26 = stablehlo.subtract %23, %25 : tensor<1x8x4xf32>
    %27 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %28 = call @_where_0(%arg0, %arg3, %18, %27) : (tensor<i64>, tensor<?x8x4xi1>, tensor<?x8x4xf32>, tensor<f32>) -> tensor<?x8x4xf32>
    %29 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %30 = stablehlo.reduce(%28 init: %29) across dimensions = [0] : (tensor<?x8x4xf32>, tensor<f32>) -> tensor<8x4xf32>
     reducer(%arg4: tensor<f32>, %arg5: tensor<f32>)  {
      %33 = stablehlo.add %arg4, %arg5 : tensor<f32>
      stablehlo.return %33 : tensor<f32>
    }
    %31 = stablehlo.broadcast_in_dim %30, dims = [1, 2] : (tensor<8x4xf32>) -> tensor<1x8x4xf32>
    %32 = stablehlo.divide %31, %26 : tensor<1x8x4xf32>
    return %32 : tensor<1x8x4xf32>
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
  func.func private @_where_0(%arg0: tensor<i64>, %arg1: tensor<?x8x4xi1>, %arg2: tensor<?x8x4xf32>, %arg3: tensor<f32>) -> tensor<?x8x4xf32> {
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

