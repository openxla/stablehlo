// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3x2xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<2xf32>
    %1 = call @expected() : () -> tensor<3x2xf32>
    %2 = stablehlo.broadcast_in_dim %0, dims = [1] : (tensor<2xf32>) -> tensor<3x2xf32>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<3x2xf32>, tensor<3x2xf32>) -> ()
    return %2 : tensor<3x2xf32>
  }
  func.func private @inputs() -> (tensor<2xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[-2.67044806, -1.32077825]> : tensor<2xf32>
    return %cst : tensor<2xf32>
  }
  func.func private @expected() -> (tensor<3x2xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-2.67044806, -1.32077825], [-2.67044806, -1.32077825], [-2.67044806, -1.32077825]]> : tensor<3x2xf32>
    return %cst : tensor<3x2xf32>
  }
}
