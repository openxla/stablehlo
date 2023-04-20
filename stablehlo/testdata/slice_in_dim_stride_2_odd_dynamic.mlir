// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x4xf32> {mhlo.sharding = ""}) -> tensor<?x4xf32> {
    %0 = stablehlo.constant dense<0> : tensor<1xi32>
    %1 = stablehlo.constant dense<0> : tensor<1xi32>
    %2 = stablehlo.concatenate %0, %1, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %3 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %4 = stablehlo.reshape %3 : (tensor<i32>) -> tensor<1xi32>
    %5 = stablehlo.constant dense<4> : tensor<1xi32>
    %6 = stablehlo.concatenate %4, %5, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %7 = stablehlo.constant dense<2> : tensor<1xi32>
    %8 = stablehlo.constant dense<1> : tensor<1xi32>
    %9 = stablehlo.concatenate %7, %8, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %10 = stablehlo.real_dynamic_slice %arg1, %2, %6, %9 : (tensor<?x4xf32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<?x4xf32>
    return %10 : tensor<?x4xf32>
  }
}

