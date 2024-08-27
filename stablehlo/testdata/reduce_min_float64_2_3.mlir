// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3xf64> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<2x3xf64>
    %1 = call @expected() : () -> tensor<3xf64>
    %cst = stablehlo.constant dense<0x7FF0000000000000> : tensor<f64>
    %2 = stablehlo.reduce(%0 init: %cst) applies stablehlo.minimum across dimensions = [0] : (tensor<2x3xf64>, tensor<f64>) -> tensor<3xf64>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<3xf64>, tensor<3xf64>) -> ()
    return %2 : tensor<3xf64>
  }
  func.func private @inputs() -> (tensor<2x3xf64> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[3.9926342288801289, 0.91457719466419918, -2.7168483023137613], [1.9138261193553205, -6.443799998513585, 5.465444979227299]]> : tensor<2x3xf64>
    return %cst : tensor<2x3xf64>
  }
  func.func private @expected() -> (tensor<3xf64> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[1.9138261193553205, -6.443799998513585, -2.7168483023137613]> : tensor<3xf64>
    return %cst : tensor<3xf64>
  }
}
