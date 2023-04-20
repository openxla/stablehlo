// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x1xf32> {mhlo.sharding = ""}) -> tensor<?x?xf64> {
    %0 = stablehlo.constant dense<2> : tensor<i64>
    %1 = stablehlo.add %arg0, %0 : tensor<i64>
    %2 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %3 = stablehlo.reshape %2 : (tensor<i32>) -> tensor<1xi32>
    %4 = stablehlo.convert %1 : (tensor<i64>) -> tensor<i32>
    %5 = stablehlo.reshape %4 : (tensor<i32>) -> tensor<1xi32>
    %6 = stablehlo.concatenate %3, %5, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %7 = stablehlo.dynamic_iota %6, dim = 0 : (tensor<2xi32>) -> tensor<?x?xi32>
    %8 = stablehlo.constant dense<0> : tensor<i32>
    %9 = stablehlo.constant dense<2> : tensor<i64>
    %10 = stablehlo.add %arg0, %9 : tensor<i64>
    %11 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %12 = stablehlo.reshape %11 : (tensor<i32>) -> tensor<1xi32>
    %13 = stablehlo.convert %10 : (tensor<i64>) -> tensor<i32>
    %14 = stablehlo.reshape %13 : (tensor<i32>) -> tensor<1xi32>
    %15 = stablehlo.concatenate %12, %14, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %16 = stablehlo.dynamic_broadcast_in_dim %8, %15, dims = [] : (tensor<i32>, tensor<2xi32>) -> tensor<?x?xi32>
    %17 = stablehlo.add %7, %16 : tensor<?x?xi32>
    %18 = stablehlo.constant dense<2> : tensor<i64>
    %19 = stablehlo.add %arg0, %18 : tensor<i64>
    %20 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %21 = stablehlo.reshape %20 : (tensor<i32>) -> tensor<1xi32>
    %22 = stablehlo.convert %19 : (tensor<i64>) -> tensor<i32>
    %23 = stablehlo.reshape %22 : (tensor<i32>) -> tensor<1xi32>
    %24 = stablehlo.concatenate %21, %23, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %25 = stablehlo.dynamic_iota %24, dim = 1 : (tensor<2xi32>) -> tensor<?x?xi32>
    %26 = stablehlo.compare  EQ, %17, %25,  SIGNED : (tensor<?x?xi32>, tensor<?x?xi32>) -> tensor<?x?xi1>
    %27 = stablehlo.convert %26 : (tensor<?x?xi1>) -> tensor<?x?xf64>
    %28 = stablehlo.convert %arg1 : (tensor<?x1xf32>) -> tensor<?x1xf64>
    %29 = stablehlo.constant dense<2> : tensor<i64>
    %30 = stablehlo.add %arg0, %29 : tensor<i64>
    %31 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %32 = stablehlo.reshape %31 : (tensor<i32>) -> tensor<1xi32>
    %33 = stablehlo.convert %30 : (tensor<i64>) -> tensor<i32>
    %34 = stablehlo.reshape %33 : (tensor<i32>) -> tensor<1xi32>
    %35 = stablehlo.concatenate %32, %34, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %36 = stablehlo.dynamic_broadcast_in_dim %28, %35, dims = [0, 1] : (tensor<?x1xf64>, tensor<2xi32>) -> tensor<?x?xf64>
    %37 = stablehlo.add %27, %36 : tensor<?x?xf64>
    return %37 : tensor<?x?xf64>
  }
}

