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
      %3 = stablehlo.add %arg0, %arg1 : tensor<ui8>
      stablehlo.return %3 : tensor<ui8>
    }) : (tensor<5x6x7xui8>, tensor<2x2x2xi64>, tensor<5x2x2xui8>) -> tensor<5x6x7xui8>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<5x6x7xui8>, tensor<5x6x7xui8>) -> ()
    return %2 : tensor<5x6x7xui8>
  }
  func.func private @inputs() -> (tensor<5x6x7xui8> {mhlo.layout_mode = "default"}, tensor<5x2x2xui8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x020500070101020504010202010205010101010400030001000202020102000004010004060002010203020200000205010001030202040300000205040300000301040002010102020102050101000001000101040206040200020100030000020001000401020102030405010101000100000602000000030304030200040002050002000001000102020107000206000000040103030301000300030204000400020105030200050703020100010102000302030203070003000001010303010300010002040201040203010500040203"> : tensor<5x6x7xui8>
    %c_0 = stablehlo.constant dense<[[[0, 1], [1, 0]], [[0, 6], [0, 0]], [[1, 0], [3, 2]], [[9, 2], [4, 7]], [[2, 0], [0, 5]]]> : tensor<5x2x2xui8>
    return %c, %c_0 : tensor<5x6x7xui8>, tensor<5x2x2xui8>
  }
  func.func private @expected() -> (tensor<5x6x7xui8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x02050007010102050401020201020501010201040003000100020202020200000401000406000201020302020000020501000103020204030000020B040300000301040002010102020102050101000001000101040306040200020100050000020001000401020102030405010101000400000602000000030304030200040902050002000001070102020107000208000000040103030301000700030204000400020105030200050903020100010102050302030203070003000001010303010300010002040201040203010500040203"> : tensor<5x6x7xui8>
    return %c : tensor<5x6x7xui8>
  }
}
