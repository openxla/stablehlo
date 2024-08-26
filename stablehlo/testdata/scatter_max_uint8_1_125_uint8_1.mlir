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
      %3 = stablehlo.maximum %arg0, %arg1 : tensor<ui8>
      stablehlo.return %3 : tensor<ui8>
    }) : (tensor<1x125xui8>, tensor<1xi64>, tensor<1xui8>) -> tensor<1x125xui8>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<1x125xui8>, tensor<1x125xui8>) -> ()
    return %2 : tensor<1x125xui8>
  }
  func.func private @inputs() -> (tensor<1x125xui8> {mhlo.layout_mode = "default"}, tensor<1xui8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x0102030406010100000105000103010003010200010408010001010200010301010002010103000304000201010001030401050000010002000101010304010003000202020200020203050303020202050102010700000200000000020101060100070000030102000200020402000301010100000300030004000401"> : tensor<1x125xui8>
    %c_0 = stablehlo.constant dense<0> : tensor<1xui8>
    return %c, %c_0 : tensor<1x125xui8>, tensor<1xui8>
  }
  func.func private @expected() -> (tensor<1x125xui8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x0102030406010100000105000103010003010200010408010001010200010301010002010103000304000201010001030401050000010002000101010304010003000202020200020203050303020202050102010700000200000000020101060100070000030102000200020402000301010100000300030004000401"> : tensor<1x125xui8>
    return %c : tensor<1x125xui8>
  }
}
