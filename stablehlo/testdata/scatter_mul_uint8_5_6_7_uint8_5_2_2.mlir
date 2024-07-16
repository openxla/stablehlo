// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<5x6x7xui8> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[[0, 1], [2, 3]], [[4, 0], [1, 2]]]> : tensor<2x2x2xi64>
    %0:2 = call @inputs() : () -> (tensor<5x6x7xui8>, tensor<5x2x2xui8>)
    %1 = call @expected() : () -> tensor<5x6x7xui8>
    %2 = "stablehlo.scatter"(%0#0, %c, %0#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1, 2], scatter_dims_to_operand_dims = [1, 2], index_vector_dim = 2>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<ui8>, %arg1: tensor<ui8>):
      %3 = stablehlo.multiply %arg0, %arg1 : tensor<ui8>
      stablehlo.return %3 : tensor<ui8>
    }) : (tensor<5x6x7xui8>, tensor<2x2x2xi64>, tensor<5x2x2xui8>) -> tensor<5x6x7xui8>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<5x6x7xui8>, tensor<5x6x7xui8>) -> ()
    return %2 : tensor<5x6x7xui8>
  }
  func.func private @inputs() -> (tensor<5x6x7xui8> {mhlo.layout_mode = "default"}, tensor<5x2x2xui8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x030002070200040002050300030001020500020003030002040005000200000501020105010000000702050302000003010102000300000105000000020001010302060000010000040301040202000200010100020502020203000103050304010300040402060000020404030102020500010202000103050006050201010503000201010200010005030302010002030500050000000102020103040105040102000005010102000104010203010504030102000400000102000100030101000400000306020005070102010300020000"> : tensor<5x6x7xui8>
    %c_0 = stablehlo.constant dense<[[[5, 0], [1, 3]], [[1, 0], [0, 2]], [[1, 1], [4, 0]], [[1, 4], [1, 2]], [[5, 3], [0, 1]]]> : tensor<5x2x2xui8>
    return %c, %c_0 : tensor<5x6x7xui8>, tensor<5x2x2xui8>
  }
  func.func private @expected() -> (tensor<5x6x7xui8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x0300020702000400020F0300030001020500020003030002040005000200000501020105010000000702050302000003010102000300000105000000020001010302060000010000040301040202000200010100020502020203000103000304010300040402060000020404030102021400010202000103050006050201010503000201010200020005030302010008030500050000000102020103040105040102000005010102000504010203010504030102000400000106000100030101000400000006020005070102010300020000"> : tensor<5x6x7xui8>
    return %c : tensor<5x6x7xui8>
  }
}
