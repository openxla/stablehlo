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
      %3 = stablehlo.maximum %arg0, %arg1 : tensor<ui8>
      stablehlo.return %3 : tensor<ui8>
    }) : (tensor<4x2x3x5xui8>, tensor<2xi64>, tensor<4x3xui8>) -> tensor<4x2x3x5xui8>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<4x2x3x5xui8>, tensor<4x2x3x5xui8>) -> ()
    return %2 : tensor<4x2x3x5xui8>
  }
  func.func private @inputs() -> (tensor<4x2x3x5xui8> {mhlo.layout_mode = "default"}, tensor<4x3xui8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x00030A000001020202050100020002020101060200000003000103020603000205040100000002000000000503020100000104020700000403010203000201020002000300050003010401020004020000040100000204010106060101020002000103010201010104070400020005030004020201020200"> : tensor<4x2x3x5xui8>
    %c_0 = stablehlo.constant dense<[[1, 2, 0], [4, 2, 2], [1, 0, 0], [1, 5, 3]]> : tensor<4x3xui8>
    return %c, %c_0 : tensor<4x2x3x5xui8>, tensor<4x3xui8>
  }
  func.func private @expected() -> (tensor<4x2x3x5xui8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x00030A000101020202050100020002020101060200000003000103020603000205040400000002020000000503020100000104020700000403010203000201020102000300050003010401020004020000040100000204010106060101020102000103050201010104070400020005030004020201020200"> : tensor<4x2x3x5xui8>
    return %c : tensor<4x2x3x5xui8>
  }
}
