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
      %3 = stablehlo.multiply %arg0, %arg1 : tensor<i16>
      stablehlo.return %3 : tensor<i16>
    }) : (tensor<4x2x3x5xi16>, tensor<2xi64>, tensor<4x3xi16>) -> tensor<4x2x3x5xi16>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<4x2x3x5xi16>, tensor<4x2x3x5xi16>) -> ()
    return %2 : tensor<4x2x3x5xi16>
  }
  func.func private @inputs() -> (tensor<4x2x3x5xi16> {mhlo.layout_mode = "default"}, tensor<4x3xi16> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x02000000FFFFFFFF0000020004000300000001000000FEFF02000100000000000000020003000100F7FFFCFF0000FCFFFFFF020000000300FEFFFFFF0300FAFF020003000500FFFFFAFF00000300FFFF0000FFFFFCFFFAFFFAFFFAFFFAFFFFFF010001000500FEFF0000FDFFFFFF00000100FDFFFCFF03000000FEFF0000F8FFFDFF00000000FCFF00000000FEFF0000F7FFFEFF0200000000000100FDFFFEFF0000FEFFFCFF010003000100FEFF020001000100FDFF000005000000000000000000FFFFFEFFFBFF0500FFFFFFFF010003000000FBFF00000200FEFF0000FDFFFFFFFFFF0100FCFF00000100FDFF0100"> : tensor<4x2x3x5xi16>
    %c_0 = stablehlo.constant dense<[[6, 2, -2], [-1, -2, -3], [2, 0, 4], [-4, 0, 2]]> : tensor<4x3xi16>
    return %c, %c_0 : tensor<4x2x3x5xi16>, tensor<4x3xi16>
  }
  func.func private @expected() -> (tensor<4x2x3x5xi16> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x02000000FFFFFFFF0000020004000300000002000000FEFF02000100000000000000020003000100F7FFFCFF0000FCFFFFFF020000000300FEFFFFFF0300FAFF02000300FBFFFFFFFAFF0000030002000000FFFFFCFFFAFF1200FAFFFAFFFFFF010001000500FEFF0000FDFFFFFF00000100FDFFFCFF03000000FEFF0000F8FFFAFF00000000FCFF00000000FEFF0000F7FFFEFF0800000000000100FDFFFEFF0000FEFFFCFF010003000100FEFF020001000100FDFF000005000000000000000000FFFFFEFF00000500FFFFFFFF010006000000FBFF00000200FEFF0000FDFFFFFFFFFF0100FCFF00000100FDFF0100"> : tensor<4x2x3x5xi16>
    return %c : tensor<4x2x3x5xi16>
  }
}
