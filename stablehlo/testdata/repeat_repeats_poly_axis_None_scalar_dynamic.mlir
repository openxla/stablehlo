// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<f32> {mhlo.sharding = ""}, %arg2: tensor<?x1xf32> {mhlo.sharding = ""}) -> tensor<?x?xf32> {
    %0 = stablehlo.reshape %arg1 : (tensor<f32>) -> tensor<1xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0] : (tensor<1xf32>) -> tensor<1x1xf32>
    %2 = stablehlo.reshape %1 : (tensor<1x1xf32>) -> tensor<1x1x1x1xf32>
    %3 = stablehlo.reshape %2 : (tensor<1x1x1x1xf32>) -> tensor<1x1x1xf32>
    %4 = stablehlo.constant dense<1> : tensor<1xi32>
    %5 = stablehlo.constant dense<1> : tensor<1xi32>
    %6 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %7 = stablehlo.reshape %6 : (tensor<i32>) -> tensor<1xi32>
    %8 = stablehlo.constant dense<1> : tensor<1xi32>
    %9 = stablehlo.concatenate %4, %5, %7, %8, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<4xi32>
    %10 = stablehlo.dynamic_broadcast_in_dim %3, %9, dims = [0, 1, 3] : (tensor<1x1x1xf32>, tensor<4xi32>) -> tensor<1x1x?x1xf32>
    %11 = stablehlo.constant dense<1> : tensor<1xi32>
    %12 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %13 = stablehlo.reshape %12 : (tensor<i32>) -> tensor<1xi32>
    %14 = stablehlo.concatenate %11, %13, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %15 = stablehlo.dynamic_reshape %10, %14 : (tensor<1x1x?x1xf32>, tensor<2xi32>) -> tensor<1x?xf32>
    %16 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %17 = stablehlo.reshape %16 : (tensor<i32>) -> tensor<1xi32>
    %18 = stablehlo.dynamic_reshape %15, %17 : (tensor<1x?xf32>, tensor<1xi32>) -> tensor<?xf32>
    %19 = stablehlo.constant dense<1> : tensor<1xi32>
    %20 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %21 = stablehlo.reshape %20 : (tensor<i32>) -> tensor<1xi32>
    %22 = stablehlo.concatenate %19, %21, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %23 = stablehlo.dynamic_broadcast_in_dim %18, %22, dims = [1] : (tensor<?xf32>, tensor<2xi32>) -> tensor<1x?xf32>
    %24 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %25 = stablehlo.reshape %24 : (tensor<i32>) -> tensor<1xi32>
    %26 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %27 = stablehlo.reshape %26 : (tensor<i32>) -> tensor<1xi32>
    %28 = stablehlo.concatenate %25, %27, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %29 = stablehlo.dynamic_broadcast_in_dim %23, %28, dims = [0, 1] : (tensor<1x?xf32>, tensor<2xi32>) -> tensor<?x?xf32>
    %30 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %31 = stablehlo.reshape %30 : (tensor<i32>) -> tensor<1xi32>
    %32 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %33 = stablehlo.reshape %32 : (tensor<i32>) -> tensor<1xi32>
    %34 = stablehlo.concatenate %31, %33, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %35 = stablehlo.dynamic_broadcast_in_dim %arg2, %34, dims = [0, 1] : (tensor<?x1xf32>, tensor<2xi32>) -> tensor<?x?xf32>
    %36 = stablehlo.add %29, %35 : tensor<?x?xf32>
    return %36 : tensor<?x?xf32>
  }
}

