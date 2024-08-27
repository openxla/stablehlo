// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<6xbf16> {jax.result_info = "[0]", mhlo.layout_mode = "default"}, tensor<6xi32> {jax.result_info = "[1]", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<4x6xbf16>, tensor<4x6xi32>)
    %1:2 = call @expected() : () -> (tensor<6xbf16>, tensor<6xi32>)
    %cst = stablehlo.constant dense<3.000000e+00> : tensor<bf16>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %2:2 = stablehlo.reduce(%0#0 init: %cst), (%0#1 init: %c) across dimensions = [0] : (tensor<4x6xbf16>, tensor<4x6xi32>, tensor<bf16>, tensor<i32>) -> (tensor<6xbf16>, tensor<6xi32>)
     reducer(%arg0: tensor<bf16>, %arg2: tensor<bf16>) (%arg1: tensor<i32>, %arg3: tensor<i32>)  {
      %3 = stablehlo.maximum %arg0, %arg2 : tensor<bf16>
      %4 = stablehlo.minimum %arg1, %arg3 : tensor<i32>
      stablehlo.return %3, %4 : tensor<bf16>, tensor<i32>
    }
    stablehlo.custom_call @check.expect_close(%2#0, %1#0) {has_side_effect = true} : (tensor<6xbf16>, tensor<6xbf16>) -> ()
    stablehlo.custom_call @check.expect_eq(%2#1, %1#1) {has_side_effect = true} : (tensor<6xi32>, tensor<6xi32>) -> ()
    return %2#0, %2#1 : tensor<6xbf16>, tensor<6xi32>
  }
  func.func private @inputs() -> (tensor<4x6xbf16> {mhlo.layout_mode = "default"}, tensor<4x6xi32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[2.334590e-03, -2.871090e-01, 1.953130e+00, 6.625000e+00, -3.234380e+00, 6.054690e-01], [2.031250e+00, -2.859380e+00, -4.593750e+00, -5.156250e+00, 2.109380e+00, -6.484380e-01], [6.062500e+00, -2.015630e+00, -5.062500e+00, 1.335940e+00, 3.015630e+00, 8.632810e-01], [2.734380e+00, 1.742190e+00, -5.812500e+00, 1.669920e-01, -3.062500e+00, -1.718750e+00]]> : tensor<4x6xbf16>
    %c = stablehlo.constant dense<[[0, 0, 0, 6, 4, 1], [3, -1, -3, -1, 0, 0], [5, 1, 6, 1, 4, 2], [0, 1, 0, -4, 2, -2]]> : tensor<4x6xi32>
    return %cst, %c : tensor<4x6xbf16>, tensor<4x6xi32>
  }
  func.func private @expected() -> (tensor<6xbf16> {mhlo.layout_mode = "default"}, tensor<6xi32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[6.062500e+00, 3.000000e+00, 3.000000e+00, 6.625000e+00, 3.015630e+00, 3.000000e+00]> : tensor<6xbf16>
    %c = stablehlo.constant dense<[0, -1, -3, -4, 0, -2]> : tensor<6xi32>
    return %cst, %c : tensor<6xbf16>, tensor<6xi32>
  }
}
