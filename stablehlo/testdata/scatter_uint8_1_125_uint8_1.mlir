// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<1x125xui8> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<0> : tensor<1xi64>
    %0:2 = call @inputs() : () -> (tensor<1x125xui8>, tensor<1xui8>)
    %1 = call @expected() : () -> tensor<1x125xui8>
    %2 = "stablehlo.scatter"(%0#0, %c, %0#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<ui8>, %arg1: tensor<ui8>):
      stablehlo.return %arg1 : tensor<ui8>
    }) : (tensor<1x125xui8>, tensor<1xi64>, tensor<1xui8>) -> tensor<1x125xui8>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<1x125xui8>, tensor<1x125xui8>) -> ()
    return %2 : tensor<1x125xui8>
  }
  func.func private @inputs() -> (tensor<1x125xui8> {mhlo.layout_mode = "default"}, tensor<1xui8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x0002020101000501040100040003030003010201030401040400010203010103000002020100010501030201020106030300020002030104000101010000050000030000010000010103040300020506030005000206000206020402000305070203000003040003020500000405010104000301000204020100010102"> : tensor<1x125xui8>
    %c_0 = stablehlo.constant dense<1> : tensor<1xui8>
    return %c, %c_0 : tensor<1x125xui8>, tensor<1xui8>
  }
  func.func private @expected() -> (tensor<1x125xui8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x0102020101000501040100040003030003010201030401040400010203010103000002020100010501030201020106030300020002030104000101010000050000030000010000010103040300020506030005000206000206020402000305070203000003040003020500000405010104000301000204020100010102"> : tensor<1x125xui8>
    return %c : tensor<1x125xui8>
  }
}
