// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x6xui16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<4x3xi8>, tensor<3x6xui16>)
    %1 = call @expected() : () -> tensor<4x6xui16>
    %2 = stablehlo.convert %0#0 : (tensor<4x3xi8>) -> tensor<4x3xui16>
    %3 = stablehlo.convert %0#1 : tensor<3x6xui16>
    %4 = stablehlo.dot_general %2, %3, contracting_dims = [1] x [0] : (tensor<4x3xui16>, tensor<3x6xui16>) -> tensor<4x6xui16>
    stablehlo.custom_call @check.expect_eq(%4, %1) {has_side_effect = true} : (tensor<4x6xui16>, tensor<4x6xui16>) -> ()
    return %4 : tensor<4x6xui16>
  }
  func.func private @inputs() -> (tensor<4x3xi8> {mhlo.layout_mode = "default"}, tensor<3x6xui16> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[-2, -2, -4], [0, -2, -3], [-4, 0, 0], [0, -1, 3]]> : tensor<4x3xi8>
    %c_0 = stablehlo.constant dense<[[0, 5, 1, 0, 2, 2], [2, 2, 1, 3, 0, 0], [0, 0, 1, 0, 1, 2]]> : tensor<3x6xui16>
    return %c, %c_0 : tensor<4x3xi8>, tensor<3x6xui16>
  }
  func.func private @expected() -> (tensor<4x6xui16> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[65532, 65522, 65528, 65530, 65528, 65524], [65532, 65532, 65531, 65530, 65533, 65530], [0, 65516, 65532, 0, 65528, 65528], [65534, 65534, 2, 65533, 3, 6]]> : tensor<4x6xui16>
    return %c : tensor<4x6xui16>
  }
}
