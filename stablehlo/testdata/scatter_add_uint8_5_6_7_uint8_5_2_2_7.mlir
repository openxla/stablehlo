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
      %3 = stablehlo.add %arg0, %arg1 : tensor<ui8>
      stablehlo.return %3 : tensor<ui8>
    }) : (tensor<5x6x7xui8>, tensor<2x2x1xi64>, tensor<5x2x2x7xui8>) -> tensor<5x6x7xui8>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<5x6x7xui8>, tensor<5x6x7xui8>) -> ()
    return %2 : tensor<5x6x7xui8>
  }
  func.func private @inputs() -> (tensor<5x6x7xui8> {mhlo.layout_mode = "default"}, tensor<5x2x2x7xui8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x000601000304020200030200000002000202020001000002030103030002020001010101000200010704020004010200000001020305060001000200010301010200050707020201030401020002040100060302000301010302020000000201000100010001000701020207000501030303010100040000020200020201010101020100020101000002030207010000050302020200040202000100010201050001000103020201000305030103000400030501030100000105000401040101030300000102040302040302020403030104"> : tensor<5x6x7xui8>
    %c_0 = stablehlo.constant dense<"0x0303010203010302000001010100000000050001050001030000000202010000010003010601000602000004010201000302070202010200000403010102020002010201050202020103000103010203020103000400020002010303010003010000000300030305040605050008020100000304030102030000010004030000020003030300070302010501"> : tensor<5x2x2x7xui8>
    return %c, %c_0 : tensor<5x6x7xui8>, tensor<5x2x2x7xui8>
  }
  func.func private @expected() -> (tensor<5x6x7xui8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x03090202060505040003030101000200020702010600010503010305000202000101010100020001070404010401030003010703030B08000104030202030403090207080902020103040102000204010006030200070402040404000201040205030203010400080403040A020604030303010100040000020200020201050103020301050402000303030207040003080806080705040A04010100010201050001000103020201000308070404020700030601070400000305030704040804050405010102040302040302020403030104"> : tensor<5x6x7xui8>
    return %c : tensor<5x6x7xui8>
  }
}
