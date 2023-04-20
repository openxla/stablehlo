// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x3x4x5xf32> {mhlo.sharding = ""}) -> tensor<?x3x20xf32> {
    %0 = stablehlo.transpose %arg1, dims = [0, 3, 1, 2] : (tensor<?x3x4x5xf32>) -> tensor<?x5x3x4xf32>
    %1 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %2 = stablehlo.reshape %1 : (tensor<i32>) -> tensor<1xi32>
    %3 = stablehlo.constant dense<3> : tensor<1xi32>
    %4 = stablehlo.constant dense<20> : tensor<1xi32>
    %5 = stablehlo.concatenate %2, %3, %4, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %6 = stablehlo.dynamic_reshape %0, %5 : (tensor<?x5x3x4xf32>, tensor<3xi32>) -> tensor<?x3x20xf32>
    return %6 : tensor<?x3x20xf32>
  }
}

