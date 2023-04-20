// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x4xf32> {mhlo.sharding = ""}) -> tensor<?x4xf32> {
    %0 = stablehlo.constant dense<-1> : tensor<i64>
    %1 = stablehlo.add %0, %arg0 : tensor<i64>
    %2 = stablehlo.constant dense<0> : tensor<1xi32>
    %3 = stablehlo.constant dense<0> : tensor<1xi32>
    %4 = stablehlo.concatenate %2, %3, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %5 = stablehlo.convert %1 : (tensor<i64>) -> tensor<i32>
    %6 = stablehlo.reshape %5 : (tensor<i32>) -> tensor<1xi32>
    %7 = stablehlo.constant dense<4> : tensor<1xi32>
    %8 = stablehlo.concatenate %6, %7, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %9 = stablehlo.constant dense<1> : tensor<1xi32>
    %10 = stablehlo.constant dense<1> : tensor<1xi32>
    %11 = stablehlo.concatenate %9, %10, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %12 = stablehlo.real_dynamic_slice %arg1, %4, %8, %11 : (tensor<?x4xf32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<?x4xf32>
    return %12 : tensor<?x4xf32>
  }
}

