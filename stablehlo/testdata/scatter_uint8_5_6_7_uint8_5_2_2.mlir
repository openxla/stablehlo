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
      stablehlo.return %arg1 : tensor<ui8>
    }) : (tensor<5x6x7xui8>, tensor<2x2x2xi64>, tensor<5x2x2xui8>) -> tensor<5x6x7xui8>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<5x6x7xui8>, tensor<5x6x7xui8>) -> ()
    return %2 : tensor<5x6x7xui8>
  }
  func.func private @inputs() -> (tensor<5x6x7xui8> {mhlo.layout_mode = "default"}, tensor<5x2x2xui8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x010103030002010203030702000103020304000702020303020002060301050700070308000000000300000003020202000408030001000001010100010202050001050203030300020203000000050203020201010300000300030300040001010301020104010302020304020103020204000502020101020000010000010501050102030202000002010000010300030000000102010002030200010203020003000501010000020000070702030001030000010203020002060301000001040106020108030100010102000302010103"> : tensor<5x6x7xui8>
    %c_0 = stablehlo.constant dense<[[[2, 0], [3, 1]], [[1, 1], [3, 1]], [[3, 0], [0, 7]], [[3, 1], [0, 2]], [[4, 3], [5, 1]]]> : tensor<5x2x2xui8>
    return %c, %c_0 : tensor<5x6x7xui8>, tensor<5x2x2xui8>
  }
  func.func private @expected() -> (tensor<5x6x7xui8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x010203030002010203010702000103020300000702020303020002060301050700070308000000000300000103020202000408010001000001010101010202050001050203030300020203000000050203020201010300000300030300070001010301020100010302020304020103020004000502020101020000010000010301050102030202020002010000010301030000000102010002030000010203020003000501010000020400070702030001010000010203020003060301000001040106020508030100010102000302010103"> : tensor<5x6x7xui8>
    return %c : tensor<5x6x7xui8>
  }
}
