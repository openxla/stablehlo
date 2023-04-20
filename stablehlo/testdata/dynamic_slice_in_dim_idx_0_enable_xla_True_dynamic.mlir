// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x4xf32> {mhlo.sharding = ""}) -> tensor<?x4xf32> {
    %0 = stablehlo.constant dense<0> : tensor<i64>
    %1 = stablehlo.constant dense<0> : tensor<i64>
    %2 = stablehlo.convert %0 : (tensor<i64>) -> tensor<i32>
    %3 = stablehlo.reshape %2 : (tensor<i32>) -> tensor<1xi32>
    %4 = stablehlo.convert %1 : (tensor<i64>) -> tensor<i32>
    %5 = stablehlo.reshape %4 : (tensor<i32>) -> tensor<1xi32>
    %6 = stablehlo.concatenate %3, %5, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %7 = stablehlo.convert %0 : (tensor<i64>) -> tensor<i32>
    %8 = stablehlo.reshape %7 : (tensor<i32>) -> tensor<1xi32>
    %9 = stablehlo.convert %1 : (tensor<i64>) -> tensor<i32>
    %10 = stablehlo.reshape %9 : (tensor<i32>) -> tensor<1xi32>
    %11 = stablehlo.concatenate %8, %10, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %12 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %13 = stablehlo.reshape %12 : (tensor<i32>) -> tensor<1xi32>
    %14 = stablehlo.constant dense<4> : tensor<1xi32>
    %15 = stablehlo.concatenate %13, %14, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %16 = stablehlo.add %11, %15 : tensor<2xi32>
    %17 = stablehlo.constant dense<1> : tensor<1xi32>
    %18 = stablehlo.constant dense<1> : tensor<1xi32>
    %19 = stablehlo.concatenate %17, %18, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %20 = stablehlo.real_dynamic_slice %arg1, %6, %16, %19 : (tensor<?x4xf32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<?x4xf32>
    return %20 : tensor<?x4xf32>
  }
}

