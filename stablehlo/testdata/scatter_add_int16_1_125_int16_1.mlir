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
      %3 = stablehlo.add %arg0, %arg1 : tensor<i16>
      stablehlo.return %3 : tensor<i16>
    }) : (tensor<1x125xi16>, tensor<1xi64>, tensor<1xi16>) -> tensor<1x125xi16>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<1x125xi16>, tensor<1x125xi16>) -> ()
    return %2 : tensor<1x125xi16>
  }
  func.func private @inputs() -> (tensor<1x125xi16> {mhlo.layout_mode = "default"}, tensor<1xi16> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x0100FCFF000002000100010002000300010000000200FFFFFFFF0300FCFF0400010000000400FFFF0200FBFF0500FEFF0100FEFF030000000200FEFF00000200FCFFFFFF0400FCFF0300020002000400FFFFFDFFFFFF0100020000000400FFFF0100FEFF02000100FEFF000001000000FBFF04000100FCFFFCFF00000300FFFF05000400FFFF04000000FDFF0300FEFF0500FEFF01000300FEFF030001000000FEFF00000200FFFF0000FFFF0100FFFFFFFF0300FFFF00000000000003000000020001000500FEFF02000000FAFF02000100010001000200FDFF03000400FCFF00000300000005000000FDFF0000010000000100040000000000"> : tensor<1x125xi16>
    %c_0 = stablehlo.constant dense<0> : tensor<1xi16>
    return %c, %c_0 : tensor<1x125xi16>, tensor<1xi16>
  }
  func.func private @expected() -> (tensor<1x125xi16> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x0100FCFF000002000100010002000300010000000200FFFFFFFF0300FCFF0400010000000400FFFF0200FBFF0500FEFF0100FEFF030000000200FEFF00000200FCFFFFFF0400FCFF0300020002000400FFFFFDFFFFFF0100020000000400FFFF0100FEFF02000100FEFF000001000000FBFF04000100FCFFFCFF00000300FFFF05000400FFFF04000000FDFF0300FEFF0500FEFF01000300FEFF030001000000FEFF00000200FFFF0000FFFF0100FFFFFFFF0300FFFF00000000000003000000020001000500FEFF02000000FAFF02000100010001000200FDFF03000400FCFF00000300000005000000FDFF0000010000000100040000000000"> : tensor<1x125xi16>
    return %c : tensor<1x125xi16>
  }
}
