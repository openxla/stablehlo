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
      %3 = stablehlo.add %arg0, %arg1 : tensor<ui8>
      stablehlo.return %3 : tensor<ui8>
    }) : (tensor<5x6x7xui8>, tensor<2x2xi64>, tensor<2x7xui8>) -> tensor<5x6x7xui8>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<5x6x7xui8>, tensor<5x6x7xui8>) -> ()
    return %2 : tensor<5x6x7xui8>
  }
  func.func private @inputs() -> (tensor<5x6x7xui8> {mhlo.layout_mode = "default"}, tensor<2x7xui8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x02030105020102030100000105030001000001030205000302010004020401020100000201010105020302040602000202000103000300020302060204020100020207020200020602000102060301010409000200010300050101050000000101010101030004010104000304020100080001000302000507000100020502050400020402030107000003000004030B010104000600000001080100050102010101000200000002010004000001000004030200060102000502040301010002000001020101000804090002000000000602"> : tensor<5x6x7xui8>
    %c_0 = stablehlo.constant dense<[[1, 5, 5, 0, 0, 6, 3], [3, 0, 1, 1, 2, 1, 1]]> : tensor<2x7xui8>
    return %c, %c_0 : tensor<5x6x7xui8>, tensor<2x7xui8>
  }
  func.func private @expected() -> (tensor<5x6x7xui8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x0203010502010204060500010B060001000001030205000302010004020401020100000201010105020302040602000202000103000300020302060204020100020207020200020602000102060301010409000200010300050101050000000101010101030004010107000405040201080001000302000507000100020502050400020402030107000003000004030B010104000600000001080100050102010101000200000002010004000001000004030200060102000502040301010002000001020101000804090002000000000602"> : tensor<5x6x7xui8>
    return %c : tensor<5x6x7xui8>
  }
}
