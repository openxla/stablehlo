// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x4xf32> {mhlo.sharding = ""}) -> tensor<1x4xf32> {
    %0 = stablehlo.constant dense<-1> : tensor<i64>
    %1 = stablehlo.add %0, %arg0 : tensor<i64>
    %2 = stablehlo.convert %1 : (tensor<i64>) -> tensor<i32>
    %3 = stablehlo.reshape %2 : (tensor<i32>) -> tensor<1xi32>
    %4 = stablehlo.constant dense<0> : tensor<1xi32>
    %5 = stablehlo.concatenate %3, %4, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %6 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %7 = stablehlo.reshape %6 : (tensor<i32>) -> tensor<1xi32>
    %8 = stablehlo.constant dense<4> : tensor<1xi32>
    %9 = stablehlo.concatenate %7, %8, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %10 = stablehlo.constant dense<1> : tensor<1xi32>
    %11 = stablehlo.constant dense<1> : tensor<1xi32>
    %12 = stablehlo.concatenate %10, %11, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %13 = stablehlo.real_dynamic_slice %arg1, %5, %9, %12 : (tensor<?x4xf32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<1x4xf32>
    return %13 : tensor<1x4xf32>
  }
}

