// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<1x50x3xui8> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<32> : tensor<1xi64>
    %0:2 = call @inputs() : () -> (tensor<1x50x3xui8>, tensor<1x3xui8>)
    %1 = call @expected() : () -> tensor<1x50x3xui8>
    %2 = "stablehlo.scatter"(%0#0, %c, %0#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<ui8>, %arg1: tensor<ui8>):
      %3 = stablehlo.multiply %arg0, %arg1 : tensor<ui8>
      stablehlo.return %3 : tensor<ui8>
    }) : (tensor<1x50x3xui8>, tensor<1xi64>, tensor<1x3xui8>) -> tensor<1x50x3xui8>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<1x50x3xui8>, tensor<1x50x3xui8>) -> ()
    return %2 : tensor<1x50x3xui8>
  }
  func.func private @inputs() -> (tensor<1x50x3xui8> {mhlo.layout_mode = "default"}, tensor<1x3xui8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x000000040104010201050102030201000300020201020101030502010002030200000300040201020101040000010006030001030801010000050000010002030303010102000404010304030202000102010200000200010600030001030601030404000103040500000101050101020000010000010002050203010303010204010209010403000000010103030603000302050300"> : tensor<1x50x3xui8>
    %c_0 = stablehlo.constant dense<[[6, 1, 0]]> : tensor<1x3xui8>
    return %c, %c_0 : tensor<1x50x3xui8>, tensor<1x3xui8>
  }
  func.func private @expected() -> (tensor<1x50x3xui8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x000000040104010201050102030201000300020201020101030502010002030200000300040201020101040000010006030001030801010000050000010002030303010102000404010304030202000102010200000200010600030001030601120400000103040500000101050101020000010000010002050203010303010204010209010403000000010103030603000302050300"> : tensor<1x50x3xui8>
    return %c : tensor<1x50x3xui8>
  }
}
