// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x2x3x5xi16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[0, 4]> : tensor<2xi64>
    %0:2 = call @inputs() : () -> (tensor<4x2x3x5xi16>, tensor<4x3xi16>)
    %1 = call @expected() : () -> tensor<4x2x3x5xi16>
    %2 = "stablehlo.scatter"(%0#0, %c, %0#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1, 3], scatter_dims_to_operand_dims = [1, 3]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<i16>, %arg1: tensor<i16>):
      %3 = stablehlo.maximum %arg0, %arg1 : tensor<i16>
      stablehlo.return %3 : tensor<i16>
    }) : (tensor<4x2x3x5xi16>, tensor<2xi64>, tensor<4x3xi16>) -> tensor<4x2x3x5xi16>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<4x2x3x5xi16>, tensor<4x2x3x5xi16>) -> ()
    return %2 : tensor<4x2x3x5xi16>
  }
  func.func private @inputs() -> (tensor<4x2x3x5xi16> {mhlo.layout_mode = "default"}, tensor<4x3xi16> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0xFFFFFBFF0000FCFF010001000300FFFFFCFFF9FFFFFFFCFF00000000000004000400FFFFFCFF03000000FFFF000000000100FDFF0300FEFFFFFFFBFFFFFFFFFF05000200FDFFFCFFFFFFFAFF0000FEFF0800000002000300FEFFFBFF0300FDFF0000020000000200000002000100FFFFFCFFFEFF03000100FFFF000001000500FDFF0000FFFF0300FDFF05000000FEFFFEFF0100020000000100FDFF0300030001000000FFFFFDFF0100FDFF0300F9FFFFFF0400030003000700FEFFFEFF010001000400010000000100FFFFFDFF0100000000000000F7FF000005000300000003000200FAFFFAFFFEFF00000000FBFF"> : tensor<4x2x3x5xi16>
    %c_0 = stablehlo.constant dense<[[-6, 1, 0], [1, 8, 0], [5, 1, 3], [0, -5, 0]]> : tensor<4x3xi16>
    return %c, %c_0 : tensor<4x2x3x5xi16>, tensor<4x3xi16>
  }
  func.func private @expected() -> (tensor<4x2x3x5xi16> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0xFFFFFBFF0000FCFF010001000300FFFFFCFF0100FFFFFCFF00000000000004000400FFFFFCFF03000000FFFF000000000100FDFF0300FEFFFFFFFBFFFFFFFFFF050002000100FCFFFFFFFAFF0000080008000000020003000000FBFF0300FDFF0000020000000200000002000100FFFFFCFFFEFF03000100FFFF00000100050005000000FFFF0300FDFF05000000FEFFFEFF0100030000000100FDFF0300030001000000FFFFFDFF0100FDFF0300F9FFFFFF0400030003000700FEFF0000010001000400010000000100FFFFFDFF0100000000000000F7FF000005000300000003000200FAFFFAFFFEFF00000000FBFF"> : tensor<4x2x3x5xi16>
    return %c : tensor<4x2x3x5xi16>
  }
}
