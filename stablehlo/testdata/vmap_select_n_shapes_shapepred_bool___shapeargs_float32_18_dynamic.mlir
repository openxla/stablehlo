// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?xi1> {mhlo.sharding = ""}, %arg2: tensor<?x18xf32> {mhlo.sharding = ""}, %arg3: tensor<?x18xf32> {mhlo.sharding = ""}) -> tensor<?x18xf32> {
    %0 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %1 = stablehlo.reshape %0 : (tensor<i32>) -> tensor<1xi32>
    %2 = stablehlo.constant dense<18> : tensor<1xi32>
    %3 = stablehlo.concatenate %1, %2, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %4 = stablehlo.dynamic_broadcast_in_dim %arg1, %3, dims = [0] : (tensor<?xi1>, tensor<2xi32>) -> tensor<?x18xi1>
    %5 = stablehlo.select %4, %arg3, %arg2 : tensor<?x18xi1>, tensor<?x18xf32>
    return %5 : tensor<?x18xf32>
  }
}

