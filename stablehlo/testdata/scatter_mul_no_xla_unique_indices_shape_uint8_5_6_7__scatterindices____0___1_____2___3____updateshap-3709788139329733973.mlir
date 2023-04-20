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
    %0 = stablehlo.constant dense<"0x000000010004000104050201000101000201050102000405000103020000020301000101010102010000000705030300010103030304020002000001000300010000000105030005000000020302050303050107040103000000060203000000040002050004030302040003030001000301010001000000020300020003060704040203050101030003010501040303020202010000020000040402000205020300060307010300000201030300030101060501050201000305000004020200010501000502000001030301020604010100"> : tensor<5x6x7xui8>
    %1 = stablehlo.constant dense<"0x0205000000010002020700020800010100030006000002020202020101080100000101030203000103050200020205010306010004020002010100000203000207000000030307010000000701020100000300020203010500020200000502010003070503020402030104000101020303000201010001040501010200000205020103000202010004020203"> : tensor<5x2x2x7xui8>
    return %0, %1 : tensor<5x6x7xui8>, tensor<5x2x2x7xui8>
  }
  func.func private @expected() -> tensor<5x6x7xui8> {
    %0 = stablehlo.constant dense<"0x00000000000400020823000200000100000300060000080A0002060200000203010001010101020100000038050000000103060900040600040000020003000600000002000600050000000203020503030501070401000000000004150000000C000E0500000015020800000000000003010100010000000203000200030C15041400060A00000F0003000F071409060804060100000200000C0402000205020300060307010300000002030300030405060502000002000605000008040200040A02000502000001030301020604010100"> : tensor<5x6x7xui8>
    return %0 : tensor<5x6x7xui8>
  }
}

