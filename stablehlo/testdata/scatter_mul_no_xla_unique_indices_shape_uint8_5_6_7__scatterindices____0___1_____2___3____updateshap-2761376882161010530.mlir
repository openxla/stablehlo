// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<[[[0], [1]], [[2], [3]]]> : tensor<2x2x1xi32>
    %1:2 = call @inputs() : () -> (tensor<5x6x7xui8>, tensor<5x2x2x7xui8>)
    %2 = call @expected() : () -> tensor<5x6x7xui8>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<ui8>, %arg1: tensor<ui8>):
      %5 = stablehlo.multiply %arg0, %arg1 : tensor<ui8>
      stablehlo.return %5 : tensor<ui8>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 3], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 2>, unique_indices = true} : (tensor<5x6x7xui8>, tensor<2x2x1xi32>, tensor<5x2x2x7xui8>) -> tensor<5x6x7xui8>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<5x6x7xui8>, tensor<5x6x7xui8>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<5x6x7xui8>, tensor<5x2x2x7xui8>) {
    %0 = stablehlo.constant dense<"0x040001030101040400010001030301010200010100010001020101020101030000020102060500010504000400010001000200020200010601010002020000060101060101030102070102020103000302010302010003040101060002000202020000000000010103010401000101040404000102070601040101000504050100020005000401000100040400030003000303020102000500020005000000010300030100020102010002040001050106010400060906000000010001010400000300010002050101010200010005050101"> : tensor<5x6x7xui8>
    %1 = stablehlo.constant dense<"0x0200010001020000030001020100000303040406010406000501060202020500000302000100000002010105010603010005060702000201040000000205020100030001030000070302020002020000010100010101010401000101040003000203000300020204030003000103040305030402000402030400010302040200000002030401030104030002"> : tensor<5x2x2x7xui8>
    return %0, %1 : tensor<5x6x7xui8>, tensor<5x2x2x7xui8>
  }
  func.func private @expected() -> tensor<5x6x7xui8> {
    %0 = stablehlo.constant dense<"0x0800010001020000000000020300000306000406000400000A010604010103000002010206050001050400080000000300000000000002060105000C0600001E06070C00020301020701020201030003020103020400000002050C0000000002060000000000020006020000000100040404000102070601040101000504050100080000000404000300080C00090006000C09000300000F000600050000000103000301000201020500080800040A03180004000C240C000000020004010C00000900020002050101010200010005050101"> : tensor<5x6x7xui8>
    return %0 : tensor<5x6x7xui8>
  }
}

