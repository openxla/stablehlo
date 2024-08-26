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
      stablehlo.return %arg1 : tensor<i16>
    }) : (tensor<1x50x3xi16>, tensor<1xi64>, tensor<1x3xi16>) -> tensor<1x50x3xi16>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<1x50x3xi16>, tensor<1x50x3xi16>) -> ()
    return %2 : tensor<1x50x3xi16>
  }
  func.func private @inputs() -> (tensor<1x50x3xi16> {mhlo.layout_mode = "default"}, tensor<1x3xi16> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x06000000040000000100FEFF0000FEFF020004000000FFFFFFFF0500020005000000FFFF05000000FEFFF7FF0000000000000200FBFFFEFFFFFF06000000FFFF02000100FEFF01000700FEFFFCFF0400FEFF02000000FEFF0000FDFFFCFFFCFF01000000FFFFFCFF0200FFFF0000050000000000FFFF06000400FFFF0000030004000100F9FFFAFF00000200010001000200FEFFFDFF0200F9FF0200FCFF00000300FFFFFEFFFDFFFEFF050004000200FFFF01000500000002000100FEFFFEFFFDFF00000000FEFF04000100050003000200FEFFFEFF0600FCFFFFFF0000010005000200000001000100FEFFFEFF0200FDFFFCFF0000FDFFFCFF0500F9FF020007000400050003000000FEFF00000300FDFFFBFF00000100FFFFFFFFFDFFFEFF000000000200FFFF0200FEFF"> : tensor<1x50x3xi16>
    %c_0 = stablehlo.constant dense<[[-6, 1, 0]]> : tensor<1x3xi16>
    return %c, %c_0 : tensor<1x50x3xi16>, tensor<1x3xi16>
  }
  func.func private @expected() -> (tensor<1x50x3xi16> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x06000000040000000100FEFF0000FEFF020004000000FFFFFFFF0500020005000000FFFF05000000FEFFF7FF0000000000000200FBFFFEFFFFFF06000000FFFF02000100FEFF01000700FEFFFCFF0400FEFF02000000FEFF0000FDFFFCFFFCFF01000000FFFFFCFF0200FFFF0000050000000000FFFF06000400FFFF0000030004000100F9FFFAFF00000200010001000200FEFFFDFF0200F9FF0200FCFF00000300FFFFFEFFFDFFFEFF050004000200FFFF01000500000002000100FEFFFEFFFAFF01000000FEFF04000100050003000200FEFFFEFF0600FCFFFFFF0000010005000200000001000100FEFFFEFF0200FDFFFCFF0000FDFFFCFF0500F9FF020007000400050003000000FEFF00000300FDFFFBFF00000100FFFFFFFFFDFFFEFF000000000200FFFF0200FEFF"> : tensor<1x50x3xi16>
    return %c : tensor<1x50x3xi16>
  }
}
