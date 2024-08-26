// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x5xui16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<4x5xui16>
    %1 = call @expected() : () -> tensor<4x5xui16>
    %2 = stablehlo.reverse %0, dims = [0] : tensor<4x5xui16>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<4x5xui16>, tensor<4x5xui16>) -> ()
    return %2 : tensor<4x5xui16>
  }
  func.func private @inputs() -> (tensor<4x5xui16> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[0, 1, 2, 0, 6], [0, 2, 1, 1, 2], [4, 2, 1, 2, 3], [1, 1, 3, 3, 0]]> : tensor<4x5xui16>
    return %c : tensor<4x5xui16>
  }
  func.func private @expected() -> (tensor<4x5xui16> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[1, 1, 3, 3, 0], [4, 2, 1, 2, 3], [0, 2, 1, 1, 2], [0, 1, 2, 0, 6]]> : tensor<4x5xui16>
    return %c : tensor<4x5xui16>
  }
}
