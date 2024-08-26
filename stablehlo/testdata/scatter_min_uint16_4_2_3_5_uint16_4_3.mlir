// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x2x3x5xui16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[0, 4]> : tensor<2xi64>
    %0:2 = call @inputs() : () -> (tensor<4x2x3x5xui16>, tensor<4x3xui16>)
    %1 = call @expected() : () -> tensor<4x2x3x5xui16>
    %2 = "stablehlo.scatter"(%0#0, %c, %0#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1, 3], scatter_dims_to_operand_dims = [1, 3]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<ui16>, %arg1: tensor<ui16>):
      %3 = stablehlo.minimum %arg0, %arg1 : tensor<ui16>
      stablehlo.return %3 : tensor<ui16>
    }) : (tensor<4x2x3x5xui16>, tensor<2xi64>, tensor<4x3xui16>) -> tensor<4x2x3x5xui16>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<4x2x3x5xui16>, tensor<4x2x3x5xui16>) -> ()
    return %2 : tensor<4x2x3x5xui16>
  }
  func.func private @inputs() -> (tensor<4x2x3x5xui16> {mhlo.layout_mode = "default"}, tensor<4x3xui16> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x050002000300010001000100010000000100040001000100020004000000010002000100010002000100020003000000010000000100040001000200010000000600060000000100060001000000000002000100000007000100030002000400000001000200010002000100000000000400020004000100000002000200000001000100020002000600010001000300010001000500010002000100070001000000000002000300040000000400030001000000060001000300020004000100020000000000020000000000000000000300000003000000050002000100000001000500000000000200000002000300"> : tensor<4x2x3x5xui16>
    %c_0 = stablehlo.constant dense<[[3, 0, 2], [4, 5, 3], [2, 2, 4], [5, 3, 3]]> : tensor<4x3xui16>
    return %c, %c_0 : tensor<4x2x3x5xui16>, tensor<4x3xui16>
  }
  func.func private @expected() -> (tensor<4x2x3x5xui16> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x050002000300010001000100010000000100000001000100020004000000010002000100010002000100020003000000010000000100040001000200010000000600060000000100060001000000000002000100000007000100030002000400000001000200010002000100000000000400020004000100000002000200000001000100020002000600010001000300010001000400010002000100070001000000000002000300040000000400030001000000060001000300020004000100020000000000020000000000000000000300000003000000050002000100000001000500000000000200000002000300"> : tensor<4x2x3x5xui16>
    return %c : tensor<4x2x3x5xui16>
  }
}
