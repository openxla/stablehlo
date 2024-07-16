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
      %3 = stablehlo.maximum %arg0, %arg1 : tensor<ui8>
      stablehlo.return %3 : tensor<ui8>
    }) : (tensor<1x50x3xui8>, tensor<1xi64>, tensor<1x3xui8>) -> tensor<1x50x3xui8>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<1x50x3xui8>, tensor<1x50x3xui8>) -> ()
    return %2 : tensor<1x50x3xui8>
  }
  func.func private @inputs() -> (tensor<1x50x3xui8> {mhlo.layout_mode = "default"}, tensor<1x3xui8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x000302020200060103010003010302000302010504040603000402000301000500010001020300000001060002000600030003030106000102030100010401010300010000060200000003010301000502050400030306020107020200000102010301000002000501010600050403030001000204000301050201010002030201010100000302010200040005010102000304050402"> : tensor<1x50x3xui8>
    %c_0 = stablehlo.constant dense<[[0, 2, 4]]> : tensor<1x3xui8>
    return %c, %c_0 : tensor<1x50x3xui8>, tensor<1x3xui8>
  }
  func.func private @expected() -> (tensor<1x50x3xui8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x000302020200060103010003010302000302010504040603000402000301000500010001020300000001060002000600030003030106000102030100010401010300010000060200000003010301000502050400030306020107020200000102010304000002000501010600050403030001000204000301050201010002030201010100000302010200040005010102000304050402"> : tensor<1x50x3xui8>
    return %c : tensor<1x50x3xui8>
  }
}
