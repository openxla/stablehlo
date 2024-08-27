// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<5x6x7xui8> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[0, 1], [2, 3]]> : tensor<2x2xi64>
    %0:2 = call @inputs() : () -> (tensor<5x6x7xui8>, tensor<2x7xui8>)
    %1 = call @expected() : () -> tensor<5x6x7xui8>
    %2 = "stablehlo.scatter"(%0#0, %c, %0#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<ui8>, %arg1: tensor<ui8>):
      %3 = stablehlo.minimum %arg0, %arg1 : tensor<ui8>
      stablehlo.return %3 : tensor<ui8>
    }) : (tensor<5x6x7xui8>, tensor<2x2xi64>, tensor<2x7xui8>) -> tensor<5x6x7xui8>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<5x6x7xui8>, tensor<5x6x7xui8>) -> ()
    return %2 : tensor<5x6x7xui8>
  }
  func.func private @inputs() -> (tensor<5x6x7xui8> {mhlo.layout_mode = "default"}, tensor<2x7xui8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x000304000200020504000303010301000302000000010401010004010001050200010002000200000102020001070400020000070000040401010001010102010205040500030200010003000000020101010602020006000002000200020001020403000407000105020004050301040101000000000700000200000103010204020102030200020000060601000200020301010104010303020203020500030001030307000102020004060203030202010502090502000400000000010102060100000206030300030000020502010301"> : tensor<5x6x7xui8>
    %c_0 = stablehlo.constant dense<[[1, 2, 8, 1, 3, 0, 4], [3, 1, 0, 1, 0, 4, 0]]> : tensor<2x7xui8>
    return %c, %c_0 : tensor<5x6x7xui8>, tensor<2x7xui8>
  }
  func.func private @expected() -> (tensor<5x6x7xui8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x000304000200020102000103000301000302000000010401010004010001050200010002000200000102020001070400020000070000040401010001010102010205040500030200010003000000020101010602020006000002000200020001020403000407000105020000010001000101000000000700000200000103010204020102030200020000060601000200020301010104010303020203020500030001030307000102020004060203030202010502090502000400000000010102060100000206030300030000020502010301"> : tensor<5x6x7xui8>
    return %c : tensor<5x6x7xui8>
  }
}
