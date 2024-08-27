// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<1x125xi16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<0> : tensor<1xi64>
    %0:2 = call @inputs() : () -> (tensor<1x125xi16>, tensor<1xi16>)
    %1 = call @expected() : () -> tensor<1x125xi16>
    %2 = "stablehlo.scatter"(%0#0, %c, %0#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<i16>, %arg1: tensor<i16>):
      %3 = stablehlo.minimum %arg0, %arg1 : tensor<i16>
      stablehlo.return %3 : tensor<i16>
    }) : (tensor<1x125xi16>, tensor<1xi64>, tensor<1xi16>) -> tensor<1x125xi16>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<1x125xi16>, tensor<1x125xi16>) -> ()
    return %2 : tensor<1x125xi16>
  }
  func.func private @inputs() -> (tensor<1x125xi16> {mhlo.layout_mode = "default"}, tensor<1xi16> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x0500FFFF02000300FDFF01000100FDFFFFFF00000000FFFF040000000000FEFF000001000000FBFF0000010000000000FEFFFFFF070002000400030000000000FAFF0000FEFF0100FBFF00000400FEFFFFFFFDFF00000000FDFFFFFFFCFF0000FCFF0200FDFF0200FFFF0400010004000000010000000000F8FFFFFFFFFF00000000FEFF04000000FDFFFEFFFEFF00000000000002000000FEFF000003000100FEFFFEFF0000000001000000FEFFFFFF0000FEFFFAFFFDFF02000200FFFFFEFFFBFFFFFFFEFFFEFF02000200FFFFFBFF0100FFFF0300FFFF010004000300FFFF00000000040002000000FFFF020006000900FEFF0200FBFF0500"> : tensor<1x125xi16>
    %c_0 = stablehlo.constant dense<3> : tensor<1xi16>
    return %c, %c_0 : tensor<1x125xi16>, tensor<1xi16>
  }
  func.func private @expected() -> (tensor<1x125xi16> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x0300FFFF02000300FDFF01000100FDFFFFFF00000000FFFF040000000000FEFF000001000000FBFF0000010000000000FEFFFFFF070002000400030000000000FAFF0000FEFF0100FBFF00000400FEFFFFFFFDFF00000000FDFFFFFFFCFF0000FCFF0200FDFF0200FFFF0400010004000000010000000000F8FFFFFFFFFF00000000FEFF04000000FDFFFEFFFEFF00000000000002000000FEFF000003000100FEFFFEFF0000000001000000FEFFFFFF0000FEFFFAFFFDFF02000200FFFFFEFFFBFFFFFFFEFFFEFF02000200FFFFFBFF0100FFFF0300FFFF010004000300FFFF00000000040002000000FFFF020006000900FEFF0200FBFF0500"> : tensor<1x125xi16>
    return %c : tensor<1x125xi16>
  }
}
