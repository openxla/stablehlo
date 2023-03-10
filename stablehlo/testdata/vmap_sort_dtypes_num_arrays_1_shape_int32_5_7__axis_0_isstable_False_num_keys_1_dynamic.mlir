// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x5x7xi32> {mhlo.sharding = ""}) -> tensor<?x5x7xi32> {
    %0 = "stablehlo.sort"(%arg1) ({
    ^bb0(%arg2: tensor<i32>, %arg3: tensor<i32>):
      %1 = stablehlo.compare  LT, %arg2, %arg3,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
    }) {dimension = 1 : i64} : (tensor<?x5x7xi32>) -> tensor<?x5x7xi32>
    return %0 : tensor<?x5x7xi32>
  }
}

