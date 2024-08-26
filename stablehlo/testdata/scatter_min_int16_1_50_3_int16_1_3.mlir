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
      %3 = stablehlo.minimum %arg0, %arg1 : tensor<i16>
      stablehlo.return %3 : tensor<i16>
    }) : (tensor<1x50x3xi16>, tensor<1xi64>, tensor<1x3xi16>) -> tensor<1x50x3xi16>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<1x50x3xi16>, tensor<1x50x3xi16>) -> ()
    return %2 : tensor<1x50x3xi16>
  }
  func.func private @inputs() -> (tensor<1x50x3xi16> {mhlo.layout_mode = "default"}, tensor<1x3xi16> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0xFFFF0000FDFFFEFF0100FFFF00000000FEFF00000000FFFF020000000000FDFF0000FEFF000000000000000002000400FEFF02000200000000000200030000000200FCFF0300FCFF01000200010000000200FEFFFAFF0000FFFFFEFF0000FCFF0000FDFFFEFFFFFF00000000FEFFFDFFFDFF0000FDFFFFFFFFFF0600FCFF0200FEFF0000FEFFFDFFFBFF0200FFFF01000300FCFF0600000000000100FFFF00000000000000000000FFFF0200FDFFFFFF0200F8FF00000000FDFF0000FFFF0300FDFF0000FDFF00000000FFFF0200FCFF020000000300FEFFFEFFFEFFFFFFFEFFFCFFFEFF0400FFFFFEFFFFFFFCFF0000FCFFFEFFFEFF02000100010000000000000001000300FDFF030005000500FDFF02000000FEFF0100FDFF03000500030000000400FEFFFFFFFEFFFEFF"> : tensor<1x50x3xi16>
    %c_0 = stablehlo.constant dense<[[3, -5, 2]]> : tensor<1x3xi16>
    return %c, %c_0 : tensor<1x50x3xi16>, tensor<1x3xi16>
  }
  func.func private @expected() -> (tensor<1x50x3xi16> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0xFFFF0000FDFFFEFF0100FFFF00000000FEFF00000000FFFF020000000000FDFF0000FEFF000000000000000002000400FEFF02000200000000000200030000000200FCFF0300FCFF01000200010000000200FEFFFAFF0000FFFFFEFF0000FCFF0000FDFFFEFFFFFF00000000FEFFFDFFFDFF0000FDFFFFFFFFFF0600FCFF0200FEFF0000FEFFFDFFFBFF0200FFFF01000300FCFF0600000000000100FFFF00000000000000000000FFFF0200FDFFFFFF0200F8FF00000000FDFF0000FFFF0300FDFFFBFFFDFF00000000FFFF0200FCFF020000000300FEFFFEFFFEFFFFFFFEFFFCFFFEFF0400FFFFFEFFFFFFFCFF0000FCFFFEFFFEFF02000100010000000000000001000300FDFF030005000500FDFF02000000FEFF0100FDFF03000500030000000400FEFFFFFFFEFFFEFF"> : tensor<1x50x3xi16>
    return %c : tensor<1x50x3xi16>
  }
}
