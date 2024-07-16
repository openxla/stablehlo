// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x2x3x5xi8> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[0, 4]> : tensor<2xi64>
    %0:2 = call @inputs() : () -> (tensor<4x2x3x5xi8>, tensor<4x3xi8>)
    %1 = call @expected() : () -> tensor<4x2x3x5xi8>
    %2 = "stablehlo.scatter"(%0#0, %c, %0#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1, 3], scatter_dims_to_operand_dims = [1, 3]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<i8>, %arg1: tensor<i8>):
      %3 = stablehlo.add %arg0, %arg1 : tensor<i8>
      stablehlo.return %3 : tensor<i8>
    }) : (tensor<4x2x3x5xi8>, tensor<2xi64>, tensor<4x3xi8>) -> tensor<4x2x3x5xi8>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<4x2x3x5xi8>, tensor<4x2x3x5xi8>) -> ()
    return %2 : tensor<4x2x3x5xi8>
  }
  func.func private @inputs() -> (tensor<4x2x3x5xi8> {mhlo.layout_mode = "default"}, tensor<4x3xi8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x03FC01FF00FFFD0003FB00020002FCFF020002FD0500FCFC000101000000010004FF00000100040203FF0403FF0002FE0000FEFE010104FF02000100030A040500FD0102FFFF00FE0306FFFFFFFEFD0300040200FD000700FF01FF03FEFF00FD00FE0000FC000503040003040500FF000104FAFF0000FEFF"> : tensor<4x2x3x5xi8>
    %c_0 = stablehlo.constant dense<[[-4, 4, 0], [4, -2, -1], [-5, -6, 1], [4, 3, 0]]> : tensor<4x3xi8>
    return %c, %c_0 : tensor<4x2x3x5xi8>, tensor<4x3xi8>
  }
  func.func private @expected() -> (tensor<4x2x3x5xi8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x03FC01FFFCFFFD0003FF00020002FCFF020002FD0500FCFC000101000000010004FF04000100040003FF0403FE0002FE0000FEFE010104FF02000100030A0405FBFD0102FFF900FE030600FFFFFEFD0300040200FD000700FF01FF03FEFF04FD00FE0003FC000503040003040500FF000104FAFF0000FEFF"> : tensor<4x2x3x5xi8>
    return %c : tensor<4x2x3x5xi8>
  }
}
