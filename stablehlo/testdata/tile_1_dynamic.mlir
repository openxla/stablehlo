// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x2xf32> {mhlo.sharding = ""}) -> tensor<?x?xf32> {
    %0 = stablehlo.constant dense<1> : tensor<1xi32>
    %1 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %2 = stablehlo.reshape %1 : (tensor<i32>) -> tensor<1xi32>
    %3 = stablehlo.constant dense<1> : tensor<1xi32>
    %4 = stablehlo.constant dense<2> : tensor<1xi32>
    %5 = stablehlo.concatenate %0, %2, %3, %4, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<4xi32>
    %6 = stablehlo.dynamic_reshape %arg1, %5 : (tensor<?x2xf32>, tensor<4xi32>) -> tensor<1x?x1x2xf32>
    %7 = stablehlo.constant dense<1> : tensor<1xi32>
    %8 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %9 = stablehlo.reshape %8 : (tensor<i32>) -> tensor<1xi32>
    %10 = stablehlo.constant dense<2> : tensor<1xi32>
    %11 = stablehlo.concatenate %7, %9, %10, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %12 = stablehlo.dynamic_reshape %6, %11 : (tensor<1x?x1x2xf32>, tensor<3xi32>) -> tensor<1x?x2xf32>
    %13 = stablehlo.constant dense<1> : tensor<1xi32>
    %14 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %15 = stablehlo.reshape %14 : (tensor<i32>) -> tensor<1xi32>
    %16 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %17 = stablehlo.reshape %16 : (tensor<i32>) -> tensor<1xi32>
    %18 = stablehlo.constant dense<2> : tensor<1xi32>
    %19 = stablehlo.concatenate %13, %15, %17, %18, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<4xi32>
    %20 = stablehlo.dynamic_broadcast_in_dim %12, %19, dims = [0, 1, 3] : (tensor<1x?x2xf32>, tensor<4xi32>) -> tensor<1x?x?x2xf32>
    %21 = stablehlo.constant dense<2> : tensor<i64>
    %22 = stablehlo.multiply %arg0, %21 : tensor<i64>
    %23 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %24 = stablehlo.reshape %23 : (tensor<i32>) -> tensor<1xi32>
    %25 = stablehlo.convert %22 : (tensor<i64>) -> tensor<i32>
    %26 = stablehlo.reshape %25 : (tensor<i32>) -> tensor<1xi32>
    %27 = stablehlo.concatenate %24, %26, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %28 = stablehlo.dynamic_reshape %20, %27 : (tensor<1x?x?x2xf32>, tensor<2xi32>) -> tensor<?x?xf32>
    return %28 : tensor<?x?xf32>
  }
}

