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
      %3 = stablehlo.minimum %arg0, %arg1 : tensor<ui8>
      stablehlo.return %3 : tensor<ui8>
    }) : (tensor<5x6x7xui8>, tensor<2x2x2xi64>, tensor<5x2x2xui8>) -> tensor<5x6x7xui8>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<5x6x7xui8>, tensor<5x6x7xui8>) -> ()
    return %2 : tensor<5x6x7xui8>
  }
  func.func private @inputs() -> (tensor<5x6x7xui8> {mhlo.layout_mode = "default"}, tensor<5x2x2xui8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x030000030004010000010304040301000205020002040001020302020001010002030302020401030002020302030101020000020306020003000503000104020000040005000101030301040901040000020301000101020100010503020000040000000101000005060100020203000100000101040002020002020403030002010300010203060103020002000001080404000001010400010001040000050001010003030301010401010205000103010303020201010505010701000605000001070002030002010002020602020104"> : tensor<5x6x7xui8>
    %c_0 = stablehlo.constant dense<[[[0, 1], [4, 2]], [[2, 2], [3, 0]], [[1, 0], [3, 0]], [[1, 3], [1, 0]], [[1, 2], [1, 3]]]> : tensor<5x2x2xui8>
    return %c, %c_0 : tensor<5x6x7xui8>, tensor<5x2x2xui8>
  }
  func.func private @expected() -> (tensor<5x6x7xui8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x030000030004010000010304040301000201020002040001020302020001010002030302020401030002020202030101020000000306020003000502000104020000040005000101030301040901040000020301000101020100010503000000040000000100000005060100020203000100000101040002020002020403030002010300010203000103020002000001080404000001010400010001040000050001010003030301010101010205000103010303020201010502010701000605000001070002030002010002020602020104"> : tensor<5x6x7xui8>
    return %c : tensor<5x6x7xui8>
  }
}
