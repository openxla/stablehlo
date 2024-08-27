// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<5x7xi8> {jax.result_info = "[0]", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<5x7xi8>
    %1 = call @expected() : () -> tensor<5x7xi8>
    %2 = "stablehlo.sort"(%0) <{dimension = 0 : i64}> ({
    ^bb0(%arg0: tensor<i8>, %arg1: tensor<i8>):
      %3 = stablehlo.compare  LT, %arg0, %arg1,  SIGNED : (tensor<i8>, tensor<i8>) -> tensor<i1>
      stablehlo.return %3 : tensor<i1>
    }) : (tensor<5x7xi8>) -> tensor<5x7xi8>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<5x7xi8>, tensor<5x7xi8>) -> ()
    return %2 : tensor<5x7xi8>
  }
  func.func private @inputs() -> (tensor<5x7xi8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[-3, -4, 3, 3, 1, -1, -4], [-5, -2, 2, 1, 5, 1, 2], [0, 0, 0, -3, -2, -5, 5], [-1, 1, 4, -3, -2, 6, -2], [-6, 0, 0, 1, 0, -4, -2]]> : tensor<5x7xi8>
    return %c : tensor<5x7xi8>
  }
  func.func private @expected() -> (tensor<5x7xi8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[-6, -4, 0, -3, -2, -5, -4], [-5, -2, 0, -3, -2, -4, -2], [-3, 0, 2, 1, 0, -1, -2], [-1, 0, 3, 1, 1, 1, 2], [0, 1, 4, 3, 5, 6, 5]]> : tensor<5x7xi8>
    return %c : tensor<5x7xi8>
  }
}
