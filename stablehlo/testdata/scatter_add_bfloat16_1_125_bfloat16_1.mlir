// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<1x125xbf16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<0> : tensor<1xi64>
    %0:2 = call @inputs() : () -> (tensor<1x125xbf16>, tensor<1xbf16>)
    %1 = call @expected() : () -> tensor<1x125xbf16>
    %2 = "stablehlo.scatter"(%0#0, %c, %0#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<bf16>, %arg1: tensor<bf16>):
      %3 = stablehlo.add %arg0, %arg1 : tensor<bf16>
      stablehlo.return %3 : tensor<bf16>
    }) : (tensor<1x125xbf16>, tensor<1xi64>, tensor<1xbf16>) -> tensor<1x125xbf16>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<1x125xbf16>, tensor<1x125xbf16>) -> ()
    return %2 : tensor<1x125xbf16>
  }
  func.func private @inputs() -> (tensor<1x125xbf16> {mhlo.layout_mode = "default"}, tensor<1xbf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0xD6C08D3FDA3F07C17CC00240204032401B408440A93F13BF4FC0923D86BF794080C03EBFA7BF3C40C7BE6240A73F78BF2E40BEC0A43D3F3F7F3D5CC069BF77401A408AC0003FE6BFB7BF72405D3FE53FAFC047C02E40A5BE88BF44BE32408B408B40863E6EC09DBCBD3E73C0A040664017C082C070C00C405DC015BE9AC00840D33FB2BFE83F50C01F408240A740AC3F4B403DC0D53F14BF50C09BBE9F3F743FA7C0353E4FC07840F53FACBF1EBF92BF85C0404051BE48BFBF3F39C0403F4C4066BFEE3E63C0AAC0CB3F5340763F3BC03DC08440E13F90C06840BD40BEC0EBBF69C019C0403FCD3EFCBFA63D9EBF043FA4BFB4BF0C40D5BD69C0"> : tensor<1x125xbf16>
    %cst_0 = stablehlo.constant dense<2.750000e+00> : tensor<1xbf16>
    return %cst, %cst_0 : tensor<1x125xbf16>, tensor<1xbf16>
  }
  func.func private @expected() -> (tensor<1x125xbf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0x7CC08D3FDA3F07C17CC00240204032401B408440A93F13BF4FC0923D86BF794080C03EBFA7BF3C40C7BE6240A73F78BF2E40BEC0A43D3F3F7F3D5CC069BF77401A408AC0003FE6BFB7BF72405D3FE53FAFC047C02E40A5BE88BF44BE32408B408B40863E6EC09DBCBD3E73C0A040664017C082C070C00C405DC015BE9AC00840D33FB2BFE83F50C01F408240A740AC3F4B403DC0D53F14BF50C09BBE9F3F743FA7C0353E4FC07840F53FACBF1EBF92BF85C0404051BE48BFBF3F39C0403F4C4066BFEE3E63C0AAC0CB3F5340763F3BC03DC08440E13F90C06840BD40BEC0EBBF69C019C0403FCD3EFCBFA63D9EBF043FA4BFB4BF0C40D5BD69C0"> : tensor<1x125xbf16>
    return %cst : tensor<1x125xbf16>
  }
}
