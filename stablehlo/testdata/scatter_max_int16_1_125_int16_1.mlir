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
      %3 = stablehlo.maximum %arg0, %arg1 : tensor<i16>
      stablehlo.return %3 : tensor<i16>
    }) : (tensor<1x125xi16>, tensor<1xi64>, tensor<1xi16>) -> tensor<1x125xi16>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<1x125xi16>, tensor<1x125xi16>) -> ()
    return %2 : tensor<1x125xi16>
  }
  func.func private @inputs() -> (tensor<1x125xi16> {mhlo.layout_mode = "default"}, tensor<1xi16> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x030000000000FAFFFDFF00000000000005000000FDFF0300000000000200060004000000FCFFFEFFFFFFFFFFFFFF0100030000000100FEFF02000100FEFF00000000FFFFFDFFFEFF01000000FFFF000002000000FEFF0200000004000100FFFFFFFF0200FDFFFEFF040003000200010003000500FFFF000000000000FFFF020003000000FEFF0100FDFF0000FFFFFBFF03000700FFFF000005000100FDFF01000100FBFF00000100000002000200FDFF040000000600FEFF01000000020007000000FFFF01000000020000000000020002000400FEFF06000300050000000000FEFF01000100FCFF000001000000000003000000FDFF00000100"> : tensor<1x125xi16>
    %c_0 = stablehlo.constant dense<-2> : tensor<1xi16>
    return %c, %c_0 : tensor<1x125xi16>, tensor<1xi16>
  }
  func.func private @expected() -> (tensor<1x125xi16> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x030000000000FAFFFDFF00000000000005000000FDFF0300000000000200060004000000FCFFFEFFFFFFFFFFFFFF0100030000000100FEFF02000100FEFF00000000FFFFFDFFFEFF01000000FFFF000002000000FEFF0200000004000100FFFFFFFF0200FDFFFEFF040003000200010003000500FFFF000000000000FFFF020003000000FEFF0100FDFF0000FFFFFBFF03000700FFFF000005000100FDFF01000100FBFF00000100000002000200FDFF040000000600FEFF01000000020007000000FFFF01000000020000000000020002000400FEFF06000300050000000000FEFF01000100FCFF000001000000000003000000FDFF00000100"> : tensor<1x125xi16>
    return %c : tensor<1x125xi16>
  }
}
