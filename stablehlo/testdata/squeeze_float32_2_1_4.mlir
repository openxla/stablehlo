// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<2x4xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<2x1x4xf32>
    %1 = call @expected() : () -> tensor<2x4xf32>
    %2 = stablehlo.reshape %0 : (tensor<2x1x4xf32>) -> tensor<2x4xf32>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<2x4xf32>, tensor<2x4xf32>) -> ()
    return %2 : tensor<2x4xf32>
  }
  func.func private @inputs() -> (tensor<2x1x4xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[[-3.65913486, -2.68992448, 1.79780495, -2.7662487]], [[-5.10396671, -1.40102684, 1.50747836, -1.04594171]]]> : tensor<2x1x4xf32>
    return %cst : tensor<2x1x4xf32>
  }
  func.func private @expected() -> (tensor<2x4xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-3.65913486, -2.68992448, 1.79780495, -2.7662487], [-5.10396671, -1.40102684, 1.50747836, -1.04594171]]> : tensor<2x4xf32>
    return %cst : tensor<2x4xf32>
  }
}
