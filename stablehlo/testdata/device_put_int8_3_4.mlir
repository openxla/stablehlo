// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3x4xi8> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<3x4xi8>
    %1 = call @expected() : () -> tensor<3x4xi8>
    stablehlo.custom_call @check.expect_eq(%0, %1) {has_side_effect = true} : (tensor<3x4xi8>, tensor<3x4xi8>) -> ()
    return %0 : tensor<3x4xi8>
  }
  func.func private @inputs() -> (tensor<3x4xi8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[-4, 4, -2, 4], [-1, 5, 0, 0], [-3, -3, 4, -1]]> : tensor<3x4xi8>
    return %c : tensor<3x4xi8>
  }
  func.func private @expected() -> (tensor<3x4xi8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[-4, 4, -2, 4], [-1, 5, 0, 0], [-3, -3, 4, -1]]> : tensor<3x4xi8>
    return %c : tensor<3x4xi8>
  }
}
