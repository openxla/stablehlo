// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x5xi32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<4x5xi32>
    %1 = call @expected() : () -> tensor<4x5xi32>
    %2 = stablehlo.reverse %0, dims = [0] : tensor<4x5xi32>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<4x5xi32>, tensor<4x5xi32>) -> ()
    return %2 : tensor<4x5xi32>
  }
  func.func private @inputs() -> (tensor<4x5xi32> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[4, -3, 2, 1, 3], [-2, 1, -2, 5, 1], [-2, -1, -1, 4, -3], [0, 3, -1, -4, 2]]> : tensor<4x5xi32>
    return %c : tensor<4x5xi32>
  }
  func.func private @expected() -> (tensor<4x5xi32> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[0, 3, -1, -4, 2], [-2, -1, -1, 4, -3], [-2, 1, -2, 5, 1], [4, -3, 2, 1, 3]]> : tensor<4x5xi32>
    return %c : tensor<4x5xi32>
  }
}
