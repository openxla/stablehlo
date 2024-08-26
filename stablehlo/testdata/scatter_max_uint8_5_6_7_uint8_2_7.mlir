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
      %3 = stablehlo.maximum %arg0, %arg1 : tensor<ui8>
      stablehlo.return %3 : tensor<ui8>
    }) : (tensor<5x6x7xui8>, tensor<2x2xi64>, tensor<2x7xui8>) -> tensor<5x6x7xui8>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<5x6x7xui8>, tensor<5x6x7xui8>) -> ()
    return %2 : tensor<5x6x7xui8>
  }
  func.func private @inputs() -> (tensor<5x6x7xui8> {mhlo.layout_mode = "default"}, tensor<2x7xui8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x030201040100040301010101020104030303000102020105060301020103020101060604010000030202000000020405000203000003050102010001010306020401010101030103020502040004000201000005020302030000040102030402020403000402020000020100000400000302010000030004020004000203000100050001010200000004000101000100030203020502050202040000010403010102000002050201000102010205020102020201020302030503040002000102020204010001060102030402010301010700"> : tensor<5x6x7xui8>
    %c_0 = stablehlo.constant dense<[[3, 2, 10, 1, 4, 1, 2], [5, 1, 4, 2, 2, 1, 3]]> : tensor<2x7xui8>
    return %c, %c_0 : tensor<5x6x7xui8>, tensor<2x7xui8>
  }
  func.func private @expected() -> (tensor<5x6x7xui8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x0302010401000403020A0104020204030303000102020105060301020103020101060604010000030202000000020405000203000003050102010001010306020401010101030103020502040004000201000005020302030000040102030402020403000402020000050104020401030302010000030004020004000203000100050001010200000004000101000100030203020502050202040000010403010102000002050201000102010205020102020201020302030503040002000102020204010001060102030402010301010700"> : tensor<5x6x7xui8>
    return %c : tensor<5x6x7xui8>
  }
}
