// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x4x6xi32> {mhlo.sharding = ""}) -> tensor<?x6xi32> {
    %0 = stablehlo.constant dense<3> : tensor<i32>
    %1 = stablehlo.reduce(%arg1 init: %0) across dimensions = [1] : (tensor<?x4x6xi32>, tensor<i32>) -> tensor<?x6xi32>
     reducer(%arg2: tensor<i32>, %arg3: tensor<i32>)  {
      %2 = stablehlo.add %arg2, %arg3 : tensor<i32>
      stablehlo.return %2 : tensor<i32>
    }
    return %1 : tensor<?x6xi32>
  }
}

