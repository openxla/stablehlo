// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x2xf32> {mhlo.sharding = ""}) -> tensor<?x2xf32> {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %2 = stablehlo.reshape %1 : (tensor<i32>) -> tensor<1xi32>
    %3 = stablehlo.constant dense<0> : tensor<1xi32>
    %4 = stablehlo.concatenate %2, %3, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %5 = stablehlo.constant dense<2> : tensor<1xi32>
    %6 = stablehlo.constant dense<0> : tensor<1xi32>
    %7 = stablehlo.concatenate %5, %6, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %8 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %9 = stablehlo.reshape %8 : (tensor<i32>) -> tensor<1xi32>
    %10 = stablehlo.constant dense<0> : tensor<1xi32>
    %11 = stablehlo.concatenate %9, %10, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %12 = stablehlo.dynamic_pad %arg1, %0, %4, %7, %11 : (tensor<?x2xf32>, tensor<f32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<?x2xf32>
    return %12 : tensor<?x2xf32>
  }
}

