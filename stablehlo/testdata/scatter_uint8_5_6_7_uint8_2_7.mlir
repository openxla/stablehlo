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
      stablehlo.return %arg1 : tensor<ui8>
    }) : (tensor<5x6x7xui8>, tensor<2x2xi64>, tensor<2x7xui8>) -> tensor<5x6x7xui8>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<5x6x7xui8>, tensor<5x6x7xui8>) -> ()
    return %2 : tensor<5x6x7xui8>
  }
  func.func private @inputs() -> (tensor<5x6x7xui8> {mhlo.layout_mode = "default"}, tensor<2x7xui8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x030104000004020204030502000300000001000200010500050204040201000303020300010403020000040000030302000402000201040201030701000301000001020202000503000402020000030101010001030201030101000000010206000000000000000000020500020200010101030002010005010506030100050503000301000200040002050101040202000005010200020104020100020201020103020500000705010104000200020204040101030400050201000200060400010302000102010002010700020401000108"> : tensor<5x6x7xui8>
    %c_0 = stablehlo.constant dense<[[1, 1, 2, 3, 0, 2, 0], [2, 0, 1, 3, 7, 1, 0]]> : tensor<2x7xui8>
    return %c, %c_0 : tensor<5x6x7xui8>, tensor<2x7xui8>
  }
  func.func private @expected() -> (tensor<5x6x7xui8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x030104000004020101020300020000000001000200010500050204040201000303020300010403020000040000030302000402000201040201030701000301000001020202000503000402020000030101010001030201030101000000010206000000000000000000020001030701000101030002010005010506030100050503000301000200040002050101040202000005010200020104020100020201020103020500000705010104000200020204040101030400050201000200060400010302000102010002010700020401000108"> : tensor<5x6x7xui8>
    return %c : tensor<5x6x7xui8>
  }
}
