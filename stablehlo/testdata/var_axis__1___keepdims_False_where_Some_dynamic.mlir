// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x8x4xf32> {mhlo.sharding = ""}, %arg2: tensor<?x8x4xi1> {mhlo.sharding = ""}) -> tensor<?x4xf32> {
    %0 = stablehlo.constant dense<0> : tensor<i64>
    %1 = call @_var(%arg0, %arg1, %0, %arg2) : (tensor<i64>, tensor<?x8x4xf32>, tensor<i64>, tensor<?x8x4xi1>) -> tensor<?x4xf32>
    return %1 : tensor<?x4xf32>
  }
  func.func private @_var(%arg0: tensor<i64>, %arg1: tensor<?x8x4xf32>, %arg2: tensor<i64>, %arg3: tensor<?x8x4xi1>) -> tensor<?x4xf32> {
    %0 = stablehlo.convert %arg3 : (tensor<?x8x4xi1>) -> tensor<?x8x4xi32>
    %1 = stablehlo.convert %0 : (tensor<?x8x4xi32>) -> tensor<?x8x4xf32>
    %2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3 = stablehlo.reduce(%1 init: %2) across dimensions = [1] : (tensor<?x8x4xf32>, tensor<f32>) -> tensor<?x4xf32>
     reducer(%arg4: tensor<f32>, %arg5: tensor<f32>)  {
      %45 = stablehlo.add %arg4, %arg5 : tensor<f32>
      stablehlo.return %45 : tensor<f32>
    }
    %4 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %5 = stablehlo.reshape %4 : (tensor<i32>) -> tensor<1xi32>
    %6 = stablehlo.constant dense<1> : tensor<1xi32>
    %7 = stablehlo.constant dense<4> : tensor<1xi32>
    %8 = stablehlo.concatenate %5, %6, %7, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %9 = stablehlo.dynamic_broadcast_in_dim %3, %8, dims = [0, 2] : (tensor<?x4xf32>, tensor<3xi32>) -> tensor<?x1x4xf32>
    %10 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %11 = call @_where(%arg0, %arg3, %arg1, %10) : (tensor<i64>, tensor<?x8x4xi1>, tensor<?x8x4xf32>, tensor<f32>) -> tensor<?x8x4xf32>
    %12 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %13 = stablehlo.reduce(%11 init: %12) across dimensions = [1] : (tensor<?x8x4xf32>, tensor<f32>) -> tensor<?x4xf32>
     reducer(%arg4: tensor<f32>, %arg5: tensor<f32>)  {
      %45 = stablehlo.add %arg4, %arg5 : tensor<f32>
      stablehlo.return %45 : tensor<f32>
    }
    %14 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %15 = stablehlo.reshape %14 : (tensor<i32>) -> tensor<1xi32>
    %16 = stablehlo.constant dense<1> : tensor<1xi32>
    %17 = stablehlo.constant dense<4> : tensor<1xi32>
    %18 = stablehlo.concatenate %15, %16, %17, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %19 = stablehlo.dynamic_broadcast_in_dim %13, %18, dims = [0, 2] : (tensor<?x4xf32>, tensor<3xi32>) -> tensor<?x1x4xf32>
    %20 = stablehlo.divide %19, %9 : tensor<?x1x4xf32>
    %21 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %22 = stablehlo.reshape %21 : (tensor<i32>) -> tensor<1xi32>
    %23 = stablehlo.constant dense<8> : tensor<1xi32>
    %24 = stablehlo.constant dense<4> : tensor<1xi32>
    %25 = stablehlo.concatenate %22, %23, %24, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %26 = stablehlo.dynamic_broadcast_in_dim %20, %25, dims = [0, 1, 2] : (tensor<?x1x4xf32>, tensor<3xi32>) -> tensor<?x8x4xf32>
    %27 = stablehlo.subtract %arg1, %26 : tensor<?x8x4xf32>
    %28 = stablehlo.multiply %27, %27 : tensor<?x8x4xf32>
    %29 = stablehlo.convert %arg3 : (tensor<?x8x4xi1>) -> tensor<?x8x4xi32>
    %30 = stablehlo.convert %29 : (tensor<?x8x4xi32>) -> tensor<?x8x4xf32>
    %31 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %32 = stablehlo.reduce(%30 init: %31) across dimensions = [1] : (tensor<?x8x4xf32>, tensor<f32>) -> tensor<?x4xf32>
     reducer(%arg4: tensor<f32>, %arg5: tensor<f32>)  {
      %45 = stablehlo.add %arg4, %arg5 : tensor<f32>
      stablehlo.return %45 : tensor<f32>
    }
    %33 = stablehlo.convert %arg2 : (tensor<i64>) -> tensor<f32>
    %34 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %35 = stablehlo.reshape %34 : (tensor<i32>) -> tensor<1xi32>
    %36 = stablehlo.constant dense<4> : tensor<1xi32>
    %37 = stablehlo.concatenate %35, %36, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %38 = stablehlo.dynamic_broadcast_in_dim %33, %37, dims = [] : (tensor<f32>, tensor<2xi32>) -> tensor<?x4xf32>
    %39 = stablehlo.subtract %32, %38 : tensor<?x4xf32>
    %40 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %41 = call @_where_0(%arg0, %arg3, %28, %40) : (tensor<i64>, tensor<?x8x4xi1>, tensor<?x8x4xf32>, tensor<f32>) -> tensor<?x8x4xf32>
    %42 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %43 = stablehlo.reduce(%41 init: %42) across dimensions = [1] : (tensor<?x8x4xf32>, tensor<f32>) -> tensor<?x4xf32>
     reducer(%arg4: tensor<f32>, %arg5: tensor<f32>)  {
      %45 = stablehlo.add %arg4, %arg5 : tensor<f32>
      stablehlo.return %45 : tensor<f32>
    }
    %44 = stablehlo.divide %43, %39 : tensor<?x4xf32>
    return %44 : tensor<?x4xf32>
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

