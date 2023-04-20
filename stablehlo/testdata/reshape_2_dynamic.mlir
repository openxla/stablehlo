// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<i64>, %arg2: tensor<i64>, %arg3: tensor<?x4x?x?x7xf32> {mhlo.sharding = ""}) -> tensor<?x28x?x?xf32> {
    %0 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %1 = stablehlo.reshape %0 : (tensor<i32>) -> tensor<1xi32>
    %2 = stablehlo.constant dense<28> : tensor<1xi32>
    %3 = stablehlo.convert %arg2 : (tensor<i64>) -> tensor<i32>
    %4 = stablehlo.reshape %3 : (tensor<i32>) -> tensor<1xi32>
    %5 = stablehlo.convert %arg1 : (tensor<i64>) -> tensor<i32>
    %6 = stablehlo.reshape %5 : (tensor<i32>) -> tensor<1xi32>
    %7 = stablehlo.concatenate %1, %2, %4, %6, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<4xi32>
    %8 = stablehlo.dynamic_reshape %arg3, %7 : (tensor<?x4x?x?x7xf32>, tensor<4xi32>) -> tensor<?x28x?x?xf32>
    return %8 : tensor<?x28x?x?xf32>
  }
}

