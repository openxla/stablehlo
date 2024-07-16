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
      %3 = stablehlo.minimum %arg0, %arg1 : tensor<ui8>
      stablehlo.return %3 : tensor<ui8>
    }) : (tensor<4x2x3x5xui8>, tensor<2xi64>, tensor<4x3xui8>) -> tensor<4x2x3x5xui8>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<4x2x3x5xui8>, tensor<4x2x3x5xui8>) -> ()
    return %2 : tensor<4x2x3x5xui8>
  }
  func.func private @inputs() -> (tensor<4x2x3x5xui8> {mhlo.layout_mode = "default"}, tensor<4x3xui8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x040400000003010001020503000001020000010401010201050102010306060001040000000006000500030307020902020200030104010201020202020202010001000002010100010304050204040103040000000303060203020200000201000002050007000205010001080106000201010300040304"> : tensor<4x2x3x5xui8>
    %c_0 = stablehlo.constant dense<[[0, 1, 3], [4, 5, 3], [3, 4, 7], [3, 0, 2]]> : tensor<4x3xui8>
    return %c, %c_0 : tensor<4x2x3x5xui8>, tensor<4x3xui8>
  }
  func.func private @expected() -> (tensor<4x2x3x5xui8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x040400000003010001010503000001020000010401010201050102010306060001040000000006000500030303020902020200030104010201020202020202010001000002010100010304050204040103040000000303060203020200000201000002000007000202010001080106000201010300040304"> : tensor<4x2x3x5xui8>
    return %c : tensor<4x2x3x5xui8>
  }
}
