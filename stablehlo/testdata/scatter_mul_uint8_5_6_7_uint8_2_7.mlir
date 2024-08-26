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
      %3 = stablehlo.multiply %arg0, %arg1 : tensor<ui8>
      stablehlo.return %3 : tensor<ui8>
    }) : (tensor<5x6x7xui8>, tensor<2x2xi64>, tensor<2x7xui8>) -> tensor<5x6x7xui8>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<5x6x7xui8>, tensor<5x6x7xui8>) -> ()
    return %2 : tensor<5x6x7xui8>
  }
  func.func private @inputs() -> (tensor<5x6x7xui8> {mhlo.layout_mode = "default"}, tensor<2x7xui8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x010000020202020103020200030603000204020201000205010300010201010303010100000100030000010000010001000301030201010303020005010102030201010003000204030704010102000502020000000102030400000301040100030201000301010202010103050003040602040104030401010203000304000202020000020203000200010101010101020001000200040102020301010002060601020005060100010101010302010103010002030200000205020102020200010701020602010400000101000000000002"> : tensor<5x6x7xui8>
    %c_0 = stablehlo.constant dense<[[2, 2, 1, 0, 2, 5, 7], [2, 3, 0, 0, 1, 4, 0]]> : tensor<2x7xui8>
    return %c, %c_0 : tensor<5x6x7xui8>, tensor<2x7xui8>
  }
  func.func private @expected() -> (tensor<5x6x7xui8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x0100000202020202060200000F2A0300020402020100020501030001020101030301010000010003000001000001000100030103020101030302000501010203020101000300020403070401010200050202000000010203040000030104010003020100030101020202030000000C000602040104030401010203000304000202020000020203000200010101010101020001000200040102020301010002060601020005060100010101010302010103010002030200000205020102020200010701020602010400000101000000000002"> : tensor<5x6x7xui8>
    return %c : tensor<5x6x7xui8>
  }
}
