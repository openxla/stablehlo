// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x2xf32> {mhlo.sharding = ""}) -> tensor<?x2xf32> {
    %0 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %1 = stablehlo.reshape %0 : (tensor<i32>) -> tensor<1xi32>
    %2 = stablehlo.constant dense<1> : tensor<1xi32>
    %3 = stablehlo.constant dense<2> : tensor<1xi32>
    %4 = stablehlo.concatenate %1, %2, %3, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %5 = stablehlo.dynamic_broadcast_in_dim %arg1, %4, dims = [0, 2] : (tensor<?x2xf32>, tensor<3xi32>) -> tensor<?x1x2xf32>
    %6 = stablehlo.constant dense<1> : tensor<1xi32>
    %7 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %8 = stablehlo.reshape %7 : (tensor<i32>) -> tensor<1xi32>
    %9 = stablehlo.constant dense<1> : tensor<1xi32>
    %10 = stablehlo.constant dense<1> : tensor<1xi32>
    %11 = stablehlo.constant dense<1> : tensor<1xi32>
    %12 = stablehlo.constant dense<2> : tensor<1xi32>
    %13 = stablehlo.concatenate %6, %8, %9, %10, %11, %12, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<6xi32>
    %14 = stablehlo.dynamic_reshape %5, %13 : (tensor<?x1x2xf32>, tensor<6xi32>) -> tensor<1x?x1x1x1x2xf32>
    %15 = stablehlo.constant dense<1> : tensor<1xi32>
    %16 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %17 = stablehlo.reshape %16 : (tensor<i32>) -> tensor<1xi32>
    %18 = stablehlo.constant dense<1> : tensor<1xi32>
    %19 = stablehlo.constant dense<1> : tensor<1xi32>
    %20 = stablehlo.constant dense<2> : tensor<1xi32>
    %21 = stablehlo.concatenate %15, %17, %18, %19, %20, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<5xi32>
    %22 = stablehlo.dynamic_reshape %14, %21 : (tensor<1x?x1x1x1x2xf32>, tensor<5xi32>) -> tensor<1x?x1x1x2xf32>
    %23 = stablehlo.constant dense<1> : tensor<1xi32>
    %24 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %25 = stablehlo.reshape %24 : (tensor<i32>) -> tensor<1xi32>
    %26 = stablehlo.constant dense<3> : tensor<1xi32>
    %27 = stablehlo.constant dense<1> : tensor<1xi32>
    %28 = stablehlo.constant dense<1> : tensor<1xi32>
    %29 = stablehlo.constant dense<2> : tensor<1xi32>
    %30 = stablehlo.concatenate %23, %25, %26, %27, %28, %29, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<6xi32>
    %31 = stablehlo.dynamic_broadcast_in_dim %22, %30, dims = [0, 1, 3, 4, 5] : (tensor<1x?x1x1x2xf32>, tensor<6xi32>) -> tensor<1x?x3x1x1x2xf32>
    %32 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %33 = stablehlo.reshape %32 : (tensor<i32>) -> tensor<1xi32>
    %34 = stablehlo.constant dense<3> : tensor<1xi32>
    %35 = stablehlo.constant dense<2> : tensor<1xi32>
    %36 = stablehlo.concatenate %33, %34, %35, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %37 = stablehlo.dynamic_reshape %31, %36 : (tensor<1x?x3x1x1x2xf32>, tensor<3xi32>) -> tensor<?x3x2xf32>
    %38 = stablehlo.constant dense<3> : tensor<i64>
    %39 = stablehlo.multiply %arg0, %38 : tensor<i64>
    %40 = stablehlo.convert %39 : (tensor<i64>) -> tensor<i32>
    %41 = stablehlo.reshape %40 : (tensor<i32>) -> tensor<1xi32>
    %42 = stablehlo.constant dense<2> : tensor<1xi32>
    %43 = stablehlo.concatenate %41, %42, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %44 = stablehlo.dynamic_reshape %37, %43 : (tensor<?x3x2xf32>, tensor<2xi32>) -> tensor<?x2xf32>
    return %44 : tensor<?x2xf32>
  }
}

