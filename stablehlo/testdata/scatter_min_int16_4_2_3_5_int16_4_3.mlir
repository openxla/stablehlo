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
      %3 = stablehlo.minimum %arg0, %arg1 : tensor<i16>
      stablehlo.return %3 : tensor<i16>
    }) : (tensor<4x2x3x5xi16>, tensor<2xi64>, tensor<4x3xi16>) -> tensor<4x2x3x5xi16>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<4x2x3x5xi16>, tensor<4x2x3x5xi16>) -> ()
    return %2 : tensor<4x2x3x5xi16>
  }
  func.func private @inputs() -> (tensor<4x2x3x5xi16> {mhlo.layout_mode = "default"}, tensor<4x3xi16> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x0000FAFF000003000000000003000600FFFF01000100FCFF02000000050002000000FFFF000000000100000000000100FFFFFBFF040000000500FDFF0000FDFF0100FEFF0200FEFFFDFFFDFF0000FFFFFFFF0000FDFFFCFF00000000FCFF0300FFFFFEFF0000030000000000FDFF0500FBFF0000020000000500FEFF01000000030000000000FFFFFDFFFDFF0100FBFF01000000FBFFFCFFFEFFFFFFFBFF0500FCFFFEFFFDFFFEFF0000FCFF040002000000050000000000FFFFFFFF0200FEFF01000300FFFFFAFFFFFF01000000FFFF00000000FCFF00000000FEFFFBFFFEFF0200FFFFFEFFFBFF0300000004000200"> : tensor<4x2x3x5xi16>
    %c_0 = stablehlo.constant dense<[[-1, -1, -1], [4, 2, -1], [3, 0, 2], [0, 0, -1]]> : tensor<4x3xi16>
    return %c, %c_0 : tensor<4x2x3x5xi16>, tensor<4x3xi16>
  }
  func.func private @expected() -> (tensor<4x2x3x5xi16> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x0000FAFF00000300FFFF000003000600FFFFFFFF0100FCFF02000000FFFF02000000FFFF000000000100000000000100FFFFFBFF040000000500FDFF0000FDFF0100FEFF0200FEFFFDFFFDFF0000FFFFFFFF0000FDFFFCFFFFFF0000FCFF0300FFFFFEFF0000030000000000FDFF0500FBFF0000020000000500FEFF01000000030000000000FFFFFDFFFDFF0100FBFF01000000FBFFFCFFFEFFFFFFFBFF0500FCFFFEFFFDFFFEFF0000FCFF040002000000050000000000FFFFFFFF0000FEFF01000300FFFFFAFFFFFF01000000FFFFFFFF0000FCFF00000000FEFFFBFFFEFF0200FFFFFEFFFBFF0300000004000200"> : tensor<4x2x3x5xi16>
    return %c : tensor<4x2x3x5xi16>
  }
}
