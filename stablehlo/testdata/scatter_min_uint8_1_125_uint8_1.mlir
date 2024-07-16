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
      %3 = stablehlo.minimum %arg0, %arg1 : tensor<ui8>
      stablehlo.return %3 : tensor<ui8>
    }) : (tensor<1x125xui8>, tensor<1xi64>, tensor<1xui8>) -> tensor<1x125xui8>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<1x125xui8>, tensor<1x125xui8>) -> ()
    return %2 : tensor<1x125xui8>
  }
  func.func private @inputs() -> (tensor<1x125xui8> {mhlo.layout_mode = "default"}, tensor<1xui8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x0003000100010404040200020001010602020102020301040102030400040101010000000104000101000003020004010300030303020102030202000604060106000000000000030102010101020001020205020302020104000302010103020002010001020200020000020102000101010001020400010306020800"> : tensor<1x125xui8>
    %c_0 = stablehlo.constant dense<0> : tensor<1xui8>
    return %c, %c_0 : tensor<1x125xui8>, tensor<1xui8>
  }
  func.func private @expected() -> (tensor<1x125xui8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x0003000100010404040200020001010602020102020301040102030400040101010000000104000101000003020004010300030303020102030202000604060106000000000000030102010101020001020205020302020104000302010103020002010001020200020000020102000101010001020400010306020800"> : tensor<1x125xui8>
    return %c : tensor<1x125xui8>
  }
}
