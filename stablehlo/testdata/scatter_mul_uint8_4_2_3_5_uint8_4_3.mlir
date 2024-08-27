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
      %3 = stablehlo.multiply %arg0, %arg1 : tensor<ui8>
      stablehlo.return %3 : tensor<ui8>
    }) : (tensor<4x2x3x5xui8>, tensor<2xi64>, tensor<4x3xui8>) -> tensor<4x2x3x5xui8>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<4x2x3x5xui8>, tensor<4x2x3x5xui8>) -> ()
    return %2 : tensor<4x2x3x5xui8>
  }
  func.func private @inputs() -> (tensor<4x2x3x5xui8> {mhlo.layout_mode = "default"}, tensor<4x3xui8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x020404030204000402020100000101000002020400010004010000020302010203000200000201000100000006000207020701010002010004010000030004020100000406010005040102030500000202000504020204010202020002000201040205050106030200010004050002000501010502030202"> : tensor<4x2x3x5xui8>
    %c_0 = stablehlo.constant dense<[[1, 0, 3], [1, 1, 0], [2, 3, 2], [5, 2, 2]]> : tensor<4x3xui8>
    return %c, %c_0 : tensor<4x2x3x5xui8>, tensor<4x3xui8>
  }
  func.func private @expected() -> (tensor<4x2x3x5xui8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x020404030204000402000100000103000002020400010004010000020302010203000200000201000100000000000207020701010002010004010000030004020200000406030005040104030500000202000504020204010202020002000A010402050A0106030200010004050002000501010502030202"> : tensor<4x2x3x5xui8>
    return %c : tensor<4x2x3x5xui8>
  }
}
