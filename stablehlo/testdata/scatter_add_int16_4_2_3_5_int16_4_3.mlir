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
      %3 = stablehlo.add %arg0, %arg1 : tensor<i16>
      stablehlo.return %3 : tensor<i16>
    }) : (tensor<4x2x3x5xi16>, tensor<2xi64>, tensor<4x3xi16>) -> tensor<4x2x3x5xi16>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<4x2x3x5xi16>, tensor<4x2x3x5xi16>) -> ()
    return %2 : tensor<4x2x3x5xi16>
  }
  func.func private @inputs() -> (tensor<4x2x3x5xi16> {mhlo.layout_mode = "default"}, tensor<4x3xi16> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0xFEFF00000100FDFF0000FBFFFFFF0000040000000100000004000000FFFFFDFF0100FDFFFFFF0000FFFF00000200FFFF0300010000000000FDFF0300FEFF03000400FEFF0100000002000000FEFF0100FEFF0300FFFFFCFFFEFFFEFFFCFF0500FEFFFEFF03000300000002000100FFFF0000010000000000FDFF0000040004000400FEFF0200FDFFFFFF0500FEFF0400030001000000FFFFFFFF040002000000FFFF00000000FFFF02000100000005000000FCFF000001000100FDFF000000000000FDFF010000000000FEFF0700FFFF050001000000040000000500FBFF0000FFFF050002000100FBFFFFFF04000100"> : tensor<4x2x3x5xi16>
    %c_0 = stablehlo.constant dense<[[-2, 3, 3], [-1, -1, -1], [0, -2, -1], [-2, 3, -2]]> : tensor<4x3xi16>
    return %c, %c_0 : tensor<4x2x3x5xi16>, tensor<4x3xi16>
  }
  func.func private @expected() -> (tensor<4x2x3x5xi16> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0xFEFF00000100FDFFFEFFFBFFFFFF00000400030001000000040000000200FDFF0100FDFFFFFF0000FFFF00000200FFFF0300010000000000FDFF0300FEFF03000400FEFF0000000002000000FEFF0000FEFF0300FFFFFCFFFDFFFEFFFCFF0500FEFFFEFF03000300000002000100FFFF0000010000000000FDFF0000040004000400FEFF0200FDFFFFFF0300FEFF040003000100FFFFFFFFFFFF040002000000FFFF00000000FFFF02000100000005000000FCFF000001000100FDFFFEFF00000000FDFF010003000000FEFF0700FFFF030001000000040000000500FBFF0000FFFF050002000100FBFFFFFF04000100"> : tensor<4x2x3x5xi16>
    return %c : tensor<4x2x3x5xi16>
  }
}
