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
      stablehlo.return %arg1 : tensor<i16>
    }) : (tensor<1x125xi16>, tensor<1xi64>, tensor<1xi16>) -> tensor<1x125xi16>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<1x125xi16>, tensor<1x125xi16>) -> ()
    return %2 : tensor<1x125xi16>
  }
  func.func private @inputs() -> (tensor<1x125xi16> {mhlo.layout_mode = "default"}, tensor<1xi16> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x00000000FEFFFCFF000007000100FBFF02000400FDFFFFFF010000000000FFFF0500000002000300060003000300FFFFFFFFFFFF000003000200020004000000FFFFFFFF0000000000000200FCFF04000100000006000000FFFF02000300FFFF0000050002000000FEFF02000000020004000100FEFFFBFF03000500FEFFFEFF0000FEFF00000100000000000100FFFFFDFFFEFF010002000400FEFF03000500FFFFFFFF030004000200FFFF0200FBFFFFFF000006000500FFFF0000FBFF01000400FDFF000000000400FEFFFFFF01000400FCFF0000FAFF0300FDFF0100020006000000FDFFFFFFFEFF01000200FDFFFFFF0200FCFFFDFF0000"> : tensor<1x125xi16>
    %c_0 = stablehlo.constant dense<0> : tensor<1xi16>
    return %c, %c_0 : tensor<1x125xi16>, tensor<1xi16>
  }
  func.func private @expected() -> (tensor<1x125xi16> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x00000000FEFFFCFF000007000100FBFF02000400FDFFFFFF010000000000FFFF0500000002000300060003000300FFFFFFFFFFFF000003000200020004000000FFFFFFFF0000000000000200FCFF04000100000006000000FFFF02000300FFFF0000050002000000FEFF02000000020004000100FEFFFBFF03000500FEFFFEFF0000FEFF00000100000000000100FFFFFDFFFEFF010002000400FEFF03000500FFFFFFFF030004000200FFFF0200FBFFFFFF000006000500FFFF0000FBFF01000400FDFF000000000400FEFFFFFF01000400FCFF0000FAFF0300FDFF0100020006000000FDFFFFFFFEFF01000200FDFFFFFF0200FCFFFDFF0000"> : tensor<1x125xi16>
    return %c : tensor<1x125xi16>
  }
}
