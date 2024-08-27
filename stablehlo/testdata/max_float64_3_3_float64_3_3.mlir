// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3x3xf64> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<3x3xf64>, tensor<3x3xf64>)
    %1 = call @expected() : () -> tensor<3x3xf64>
    %2 = stablehlo.maximum %0#0, %0#1 : tensor<3x3xf64>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<3x3xf64>, tensor<3x3xf64>) -> ()
    return %2 : tensor<3x3xf64>
  }
  func.func private @inputs() -> (tensor<3x3xf64> {mhlo.layout_mode = "default"}, tensor<3x3xf64> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[0x7FF8000000000000, 0x7FF8000000000000, 0x7FF8000000000000], [0x7FF0000000000000, 0x7FF0000000000000, 0x7FF0000000000000], [0xFFF0000000000000, 0xFFF0000000000000, 0xFFF0000000000000]]> : tensor<3x3xf64>
    %cst_0 = stablehlo.constant dense<[[0x7FF8000000000000, 0x7FF0000000000000, 0xFFF0000000000000], [0x7FF8000000000000, 0x7FF0000000000000, 0xFFF0000000000000], [0x7FF8000000000000, 0x7FF0000000000000, 0xFFF0000000000000]]> : tensor<3x3xf64>
    return %cst, %cst_0 : tensor<3x3xf64>, tensor<3x3xf64>
  }
  func.func private @expected() -> (tensor<3x3xf64> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[0x7FF8000000000000, 0x7FF8000000000000, 0x7FF8000000000000], [0x7FF8000000000000, 0x7FF0000000000000, 0x7FF0000000000000], [0x7FF8000000000000, 0x7FF0000000000000, 0xFFF0000000000000]]> : tensor<3x3xf64>
    return %cst : tensor<3x3xf64>
  }
}
