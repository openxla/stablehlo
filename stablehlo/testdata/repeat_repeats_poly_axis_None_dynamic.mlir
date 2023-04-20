// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x2xf32> {mhlo.sharding = ""}) -> tensor<?xf32> {
    %0 = stablehlo.constant dense<2> : tensor<i64>
    %1 = stablehlo.multiply %arg0, %0 : tensor<i64>
    %2 = stablehlo.convert %1 : (tensor<i64>) -> tensor<i32>
    %3 = stablehlo.reshape %2 : (tensor<i32>) -> tensor<1xi32>
    %4 = stablehlo.dynamic_reshape %arg1, %3 : (tensor<?x2xf32>, tensor<1xi32>) -> tensor<?xf32>
    %5 = stablehlo.constant dense<2> : tensor<i64>
    %6 = stablehlo.multiply %arg0, %5 : tensor<i64>
    %7 = stablehlo.convert %6 : (tensor<i64>) -> tensor<i32>
    %8 = stablehlo.reshape %7 : (tensor<i32>) -> tensor<1xi32>
    %9 = stablehlo.constant dense<1> : tensor<1xi32>
    %10 = stablehlo.concatenate %8, %9, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %11 = stablehlo.dynamic_broadcast_in_dim %4, %10, dims = [0] : (tensor<?xf32>, tensor<2xi32>) -> tensor<?x1xf32>
    %12 = stablehlo.constant dense<2> : tensor<i64>
    %13 = stablehlo.multiply %arg0, %12 : tensor<i64>
    %14 = stablehlo.constant dense<1> : tensor<1xi32>
    %15 = stablehlo.convert %13 : (tensor<i64>) -> tensor<i32>
    %16 = stablehlo.reshape %15 : (tensor<i32>) -> tensor<1xi32>
    %17 = stablehlo.constant dense<1> : tensor<1xi32>
    %18 = stablehlo.constant dense<1> : tensor<1xi32>
    %19 = stablehlo.concatenate %14, %16, %17, %18, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<4xi32>
    %20 = stablehlo.dynamic_reshape %11, %19 : (tensor<?x1xf32>, tensor<4xi32>) -> tensor<1x?x1x1xf32>
    %21 = stablehlo.constant dense<2> : tensor<i64>
    %22 = stablehlo.multiply %arg0, %21 : tensor<i64>
    %23 = stablehlo.constant dense<1> : tensor<1xi32>
    %24 = stablehlo.convert %22 : (tensor<i64>) -> tensor<i32>
    %25 = stablehlo.reshape %24 : (tensor<i32>) -> tensor<1xi32>
    %26 = stablehlo.constant dense<1> : tensor<1xi32>
    %27 = stablehlo.concatenate %23, %25, %26, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %28 = stablehlo.dynamic_reshape %20, %27 : (tensor<1x?x1x1xf32>, tensor<3xi32>) -> tensor<1x?x1xf32>
    %29 = stablehlo.constant dense<2> : tensor<i64>
    %30 = stablehlo.multiply %arg0, %29 : tensor<i64>
    %31 = stablehlo.constant dense<1> : tensor<1xi32>
    %32 = stablehlo.convert %30 : (tensor<i64>) -> tensor<i32>
    %33 = stablehlo.reshape %32 : (tensor<i32>) -> tensor<1xi32>
    %34 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %35 = stablehlo.reshape %34 : (tensor<i32>) -> tensor<1xi32>
    %36 = stablehlo.constant dense<1> : tensor<1xi32>
    %37 = stablehlo.concatenate %31, %33, %35, %36, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<4xi32>
    %38 = stablehlo.dynamic_broadcast_in_dim %28, %37, dims = [0, 1, 3] : (tensor<1x?x1xf32>, tensor<4xi32>) -> tensor<1x?x?x1xf32>
    %39 = stablehlo.constant dense<2> : tensor<i64>
    %40 = stablehlo.multiply %arg0, %39 : tensor<i64>
    %41 = stablehlo.convert %40 : (tensor<i64>) -> tensor<i32>
    %42 = stablehlo.reshape %41 : (tensor<i32>) -> tensor<1xi32>
    %43 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %44 = stablehlo.reshape %43 : (tensor<i32>) -> tensor<1xi32>
    %45 = stablehlo.concatenate %42, %44, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %46 = stablehlo.dynamic_reshape %38, %45 : (tensor<1x?x?x1xf32>, tensor<2xi32>) -> tensor<?x?xf32>
    %47 = stablehlo.multiply %arg0, %arg0 : tensor<i64>
    %48 = stablehlo.constant dense<2> : tensor<i64>
    %49 = stablehlo.multiply %47, %48 : tensor<i64>
    %50 = stablehlo.convert %49 : (tensor<i64>) -> tensor<i32>
    %51 = stablehlo.reshape %50 : (tensor<i32>) -> tensor<1xi32>
    %52 = stablehlo.dynamic_reshape %46, %51 : (tensor<?x?xf32>, tensor<1xi32>) -> tensor<?xf32>
    return %52 : tensor<?xf32>
  }
}

