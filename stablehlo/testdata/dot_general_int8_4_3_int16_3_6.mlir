// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x6xi16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<4x3xi8>, tensor<3x6xi16>)
    %1 = call @expected() : () -> tensor<4x6xi16>
    %2 = stablehlo.convert %0#0 : (tensor<4x3xi8>) -> tensor<4x3xi16>
    %3 = stablehlo.convert %0#1 : tensor<3x6xi16>
    %4 = stablehlo.dot_general %2, %3, contracting_dims = [1] x [0] : (tensor<4x3xi16>, tensor<3x6xi16>) -> tensor<4x6xi16>
    stablehlo.custom_call @check.expect_eq(%4, %1) {has_side_effect = true} : (tensor<4x6xi16>, tensor<4x6xi16>) -> ()
    return %4 : tensor<4x6xi16>
  }
  func.func private @inputs() -> (tensor<4x3xi8> {mhlo.layout_mode = "default"}, tensor<3x6xi16> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[-4, 0, 1], [2, 4, 0], [-1, 0, 1], [2, 3, -1]]> : tensor<4x3xi8>
    %c_0 = stablehlo.constant dense<[[3, -3, 1, -1, 0, -2], [-3, 4, 3, -1, -1, 0], [-4, -2, 2, 1, 0, -4]]> : tensor<3x6xi16>
    return %c, %c_0 : tensor<4x3xi8>, tensor<3x6xi16>
  }
  func.func private @expected() -> (tensor<4x6xi16> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[-16, 10, -2, 5, 0, 4], [-6, 10, 14, -6, -4, -4], [-7, 1, 1, 2, 0, -2], [1, 8, 9, -6, -3, 0]]> : tensor<4x6xi16>
    return %c : tensor<4x6xi16>
  }
}
