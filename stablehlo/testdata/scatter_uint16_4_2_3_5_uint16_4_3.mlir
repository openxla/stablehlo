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
      stablehlo.return %arg1 : tensor<ui16>
    }) : (tensor<4x2x3x5xui16>, tensor<2xi64>, tensor<4x3xui16>) -> tensor<4x2x3x5xui16>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<4x2x3x5xui16>, tensor<4x2x3x5xui16>) -> ()
    return %2 : tensor<4x2x3x5xui16>
  }
  func.func private @inputs() -> (tensor<4x2x3x5xui16> {mhlo.layout_mode = "default"}, tensor<4x3xui16> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x040000000000050002000300000001000200020000000400070000000100010006000300000003000400020003000400010005000200020000000300060000000200030004000000000001000400060002000200000001000100000002000000010000000100030001000200000000000000010000000000000001000300030000000400000005000000020004000400000003000200010000000000020000000000070002000400020001000000040001000100020001000100020000000500020000000000000005000000010001000200040000000400020004000000010000000200060003000400030003000000"> : tensor<4x2x3x5xui16>
    %c_0 = stablehlo.constant dense<[[7, 4, 1], [0, 0, 2], [5, 1, 5], [0, 3, 0]]> : tensor<4x3xui16>
    return %c, %c_0 : tensor<4x2x3x5xui16>, tensor<4x3xui16>
  }
  func.func private @expected() -> (tensor<4x2x3x5xui16> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x040000000000050007000300000001000200040000000400070000000100010006000300000003000400020003000400010005000200020000000300060000000200030000000000000001000400000002000200000001000200000002000000010000000100030001000200000000000000010000000000000001000300030005000400000005000000010004000400000003000500010000000000020000000000070002000400020001000000040001000100020001000100020000000500020000000000030005000000010001000000040000000400020004000000010000000200060003000400030003000000"> : tensor<4x2x3x5xui16>
    return %c : tensor<4x2x3x5xui16>
  }
}
