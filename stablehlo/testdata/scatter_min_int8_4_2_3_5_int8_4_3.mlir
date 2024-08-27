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
      %3 = stablehlo.minimum %arg0, %arg1 : tensor<i8>
      stablehlo.return %3 : tensor<i8>
    }) : (tensor<4x2x3x5xi8>, tensor<2xi64>, tensor<4x3xi8>) -> tensor<4x2x3x5xi8>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<4x2x3x5xi8>, tensor<4x2x3x5xi8>) -> ()
    return %2 : tensor<4x2x3x5xi8>
  }
  func.func private @inputs() -> (tensor<4x2x3x5xi8> {mhlo.layout_mode = "default"}, tensor<4x3xi8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x0002FD00FAFF00FA030104FEFD01FF04FE000002FF000200FE01FD0200FE02000501020100FFFF03FFFEFFFD00FFFBFEFDFF0001010000FD0000FBFD000002FD0103010004FE00FEFBFDFDFB01FFFF00FD00000100F9FFFE04FD000400010002FDFE01F80000000102F8FFFB02000000FFFAFC000000FD00"> : tensor<4x2x3x5xi8>
    %c_0 = stablehlo.constant dense<[[0, 2, 0], [0, 0, -1], [-1, 3, 0], [-4, 3, 1]]> : tensor<4x3xi8>
    return %c, %c_0 : tensor<4x2x3x5xi8>, tensor<4x3xi8>
  }
  func.func private @expected() -> (tensor<4x2x3x5xi8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x0002FD00FAFF00FA030104FEFD01FF04FE000002FF000200FE01FD0200FE02000501000100FFFF00FFFEFFFDFFFFFBFEFDFF0001010000FD0000FBFD000002FDFF03010004FE00FEFBFDFDFB01FFFF00FD00000100F9FFFE04FD00040001FC02FDFE01F80000000101F8FFFB02000000FFFAFC000000FD00"> : tensor<4x2x3x5xi8>
    return %c : tensor<4x2x3x5xi8>
  }
}
