// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<1x50x3xi16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<32> : tensor<1xi64>
    %0:2 = call @inputs() : () -> (tensor<1x50x3xi16>, tensor<1x3xi16>)
    %1 = call @expected() : () -> tensor<1x50x3xi16>
    %2 = "stablehlo.scatter"(%0#0, %c, %0#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<i16>, %arg1: tensor<i16>):
      %3 = stablehlo.maximum %arg0, %arg1 : tensor<i16>
      stablehlo.return %3 : tensor<i16>
    }) : (tensor<1x50x3xi16>, tensor<1xi64>, tensor<1x3xi16>) -> tensor<1x50x3xi16>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<1x50x3xi16>, tensor<1x50x3xi16>) -> ()
    return %2 : tensor<1x50x3xi16>
  }
  func.func private @inputs() -> (tensor<1x50x3xi16> {mhlo.layout_mode = "default"}, tensor<1x3xi16> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x0000030003000300000003000100FBFF00000200FBFF00000000070003000000000005000000FAFF0400030002000200FEFF00000200FDFFFEFFFAFFFFFF00000300FCFFFFFF000000000100FDFF01000400FDFF0300FFFF0100FFFFFFFF03000300FFFF00000000020001000100FCFFFCFF00000400FEFF0100FDFF0000FFFF0000FEFF0400FCFF0500FDFF0600020001000000FDFF03000100FDFF03000000FDFF04000000FBFF000000000000FDFFFDFFFDFF02000000010003000200000002000500FDFFFEFF00000100FFFF0200FEFFFEFFFFFFFCFF0100000000000300FDFFFFFF000001000000FDFFFFFFFFFF00000000FDFFFFFF0300000000000300FEFF0000FCFF05000100010000000100FDFF0100FBFF00000200FFFFFEFF030001000400FCFFFFFFFEFFFFFF"> : tensor<1x50x3xi16>
    %c_0 = stablehlo.constant dense<[[-1, -2, 1]]> : tensor<1x3xi16>
    return %c, %c_0 : tensor<1x50x3xi16>, tensor<1x3xi16>
  }
  func.func private @expected() -> (tensor<1x50x3xi16> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x0000030003000300000003000100FBFF00000200FBFF00000000070003000000000005000000FAFF0400030002000200FEFF00000200FDFFFEFFFAFFFFFF00000300FCFFFFFF000000000100FDFF01000400FDFF0300FFFF0100FFFFFFFF03000300FFFF00000000020001000100FCFFFCFF00000400FEFF0100FDFF0000FFFF0000FEFF0400FCFF0500FDFF0600020001000000FDFF03000100FDFF03000000FDFF04000000FBFF000000000000FDFFFDFFFDFF020000000100030002000000020005000100FEFF00000100FFFF0200FEFFFEFFFFFFFCFF0100000000000300FDFFFFFF000001000000FDFFFFFFFFFF00000000FDFFFFFF0300000000000300FEFF0000FCFF05000100010000000100FDFF0100FBFF00000200FFFFFEFF030001000400FCFFFFFFFEFFFFFF"> : tensor<1x50x3xi16>
    return %c : tensor<1x50x3xi16>
  }
}
