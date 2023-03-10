// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x4xf32> {mhlo.sharding = ""}, %arg2: tensor<i32> {mhlo.sharding = ""}) -> tensor<?x4xf32> {
    %0 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %1 = stablehlo.constant dense<0> : tensor<i32>
    %2 = stablehlo.compare  LT, %arg2, %1,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %3 = stablehlo.add %arg2, %0 : tensor<i32>
    %4 = stablehlo.select %2, %3, %arg2 : tensor<i1>, tensor<i32>
    %5 = stablehlo.constant dense<0> : tensor<i32>
    %6 = stablehlo.dynamic_update_slice %arg1, %arg1, %4, %5 : (tensor<?x4xf32>, tensor<?x4xf32>, tensor<i32>, tensor<i32>) -> tensor<?x4xf32>
    return %6 : tensor<?x4xf32>
  }
}

