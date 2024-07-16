// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x2x3x5xui8> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[0, 4]> : tensor<2xi64>
    %0:2 = call @inputs() : () -> (tensor<4x2x3x5xui8>, tensor<4x3xui8>)
    %1 = call @expected() : () -> tensor<4x2x3x5xui8>
    %2 = "stablehlo.scatter"(%0#0, %c, %0#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1, 3], scatter_dims_to_operand_dims = [1, 3]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<ui8>, %arg1: tensor<ui8>):
      stablehlo.return %arg1 : tensor<ui8>
    }) : (tensor<4x2x3x5xui8>, tensor<2xi64>, tensor<4x3xui8>) -> tensor<4x2x3x5xui8>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<4x2x3x5xui8>, tensor<4x2x3x5xui8>) -> ()
    return %2 : tensor<4x2x3x5xui8>
  }
  func.func private @inputs() -> (tensor<4x2x3x5xui8> {mhlo.layout_mode = "default"}, tensor<4x3xui8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x050000020402020202050002010201030201030006000101040103020104020002060003030002000102000003010704020103020202010306000403030401030304010400000000010200000504030502020304020402030200000002030704010705030104000500030000000002020300010003000101"> : tensor<4x2x3x5xui8>
    %c_0 = stablehlo.constant dense<[[1, 0, 1], [5, 4, 2], [0, 3, 1], [0, 3, 4]]> : tensor<4x3xui8>
    return %c, %c_0 : tensor<4x2x3x5xui8>, tensor<4x3xui8>
  }
  func.func private @expected() -> (tensor<4x2x3x5xui8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x050000020102020202000002010201030201030006000101040103020104020002060503030002040102000002010704020103020202010306000403030401030004010400030000010201000504030502020304020402030200000002030004010705030104000504030000000002020300010003000101"> : tensor<4x2x3x5xui8>
    return %c : tensor<4x2x3x5xui8>
  }
}
