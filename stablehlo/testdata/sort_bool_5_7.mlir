// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<5x7xi1> {jax.result_info = "[0]", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<5x7xi1>
    %1 = call @expected() : () -> tensor<5x7xi1>
    %2 = "stablehlo.sort"(%0) <{dimension = 0 : i64}> ({
    ^bb0(%arg0: tensor<i1>, %arg1: tensor<i1>):
      %3 = stablehlo.compare  LT, %arg0, %arg1,  UNSIGNED : (tensor<i1>, tensor<i1>) -> tensor<i1>
      stablehlo.return %3 : tensor<i1>
    }) : (tensor<5x7xi1>) -> tensor<5x7xi1>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<5x7xi1>, tensor<5x7xi1>) -> ()
    return %2 : tensor<5x7xi1>
  }
  func.func private @inputs() -> (tensor<5x7xi1> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<true> : tensor<5x7xi1>
    return %c : tensor<5x7xi1>
  }
  func.func private @expected() -> (tensor<5x7xi1> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<true> : tensor<5x7xi1>
    return %c : tensor<5x7xi1>
  }
}
