// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x5x3xf32> {mhlo.sharding = ""}) -> tensor<?x2x2xf32> {
    %0 = stablehlo.constant dense<0> : tensor<1xi32>
    %1 = stablehlo.constant dense<1> : tensor<1xi32>
    %2 = stablehlo.constant dense<1> : tensor<1xi32>
    %3 = stablehlo.concatenate %0, %1, %2, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %4 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %5 = stablehlo.reshape %4 : (tensor<i32>) -> tensor<1xi32>
    %6 = stablehlo.constant dense<5> : tensor<1xi32>
    %7 = stablehlo.constant dense<3> : tensor<1xi32>
    %8 = stablehlo.concatenate %5, %6, %7, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %9 = stablehlo.constant dense<1> : tensor<1xi32>
    %10 = stablehlo.constant dense<2> : tensor<1xi32>
    %11 = stablehlo.constant dense<1> : tensor<1xi32>
    %12 = stablehlo.concatenate %9, %10, %11, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %13 = stablehlo.real_dynamic_slice %arg1, %3, %8, %12 : (tensor<?x5x3xf32>, tensor<3xi32>, tensor<3xi32>, tensor<3xi32>) -> tensor<?x2x2xf32>
    return %13 : tensor<?x2x2xf32>
  }
}

