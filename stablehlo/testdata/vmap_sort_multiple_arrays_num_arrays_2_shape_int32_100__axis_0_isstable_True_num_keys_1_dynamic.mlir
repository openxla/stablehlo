// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x100xi32> {mhlo.sharding = ""}, %arg2: tensor<?x100xi32> {mhlo.sharding = ""}) -> (tensor<?x100xi32>, tensor<?x100xi32>) {
    %0:2 = "stablehlo.sort"(%arg1, %arg2) ({
    ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>, %arg5: tensor<i32>, %arg6: tensor<i32>):
      %1 = stablehlo.compare  LT, %arg3, %arg4,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
    }) {dimension = 1 : i64, is_stable = true} : (tensor<?x100xi32>, tensor<?x100xi32>) -> (tensor<?x100xi32>, tensor<?x100xi32>)
    return %0#0, %0#1 : tensor<?x100xi32>, tensor<?x100xi32>
  }
}

