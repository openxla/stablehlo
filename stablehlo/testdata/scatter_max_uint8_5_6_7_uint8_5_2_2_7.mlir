// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<5x6x7xui8> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[[0], [1]], [[2], [3]]]> : tensor<2x2x1xi64>
    %0:2 = call @inputs() : () -> (tensor<5x6x7xui8>, tensor<5x2x2x7xui8>)
    %1 = call @expected() : () -> tensor<5x6x7xui8>
    %2 = "stablehlo.scatter"(%0#0, %c, %0#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 3], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 2>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<ui8>, %arg1: tensor<ui8>):
      %3 = stablehlo.maximum %arg0, %arg1 : tensor<ui8>
      stablehlo.return %3 : tensor<ui8>
    }) : (tensor<5x6x7xui8>, tensor<2x2x1xi64>, tensor<5x2x2x7xui8>) -> tensor<5x6x7xui8>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<5x6x7xui8>, tensor<5x6x7xui8>) -> ()
    return %2 : tensor<5x6x7xui8>
  }
  func.func private @inputs() -> (tensor<5x6x7xui8> {mhlo.layout_mode = "default"}, tensor<5x2x2x7xui8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x020502010000010302040102030002030001010500000003020201010205030400040400020001050102000304020101000000020100000400030100060200010106010002010101020204020101010401000002030201050302040102030203010404060103040404010006010001020201000201000104010000000200010200050002000603030000030200000000000500030402000300010102000102030302000302000200010006010106030304000101000001010603010101030100050103020403060203010103020000010107"> : tensor<5x6x7xui8>
    %c_0 = stablehlo.constant dense<"0x0003000201050203010001020000040302000002000406020105020304000403010402010101020003010104010002000000010007000003010301020002010502050200010100050203020501010300000104040002000703040102020200000004030501030406050003050000020307010000030101060501030105010000060201000503030102020200"> : tensor<5x2x2x7xui8>
    return %c, %c_0 : tensor<5x6x7xui8>, tensor<5x2x2x7xui8>
  }
  func.func private @expected() -> (tensor<5x6x7xui8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x020502020105020302040102030004030201010500040603020502030205030400040400020001050102040304030104020101020200030401040100060200010106070002030101020204020101010401000002030301050302040502050203010404060203040504010306010104040201000201000104010000000200010200070304010603030000030403050103040605030405000302030102000102030302000302000200070106010306030605010301050101010603010105030301050203020403060203010103020000010107"> : tensor<5x6x7xui8>
    return %c : tensor<5x6x7xui8>
  }
}
