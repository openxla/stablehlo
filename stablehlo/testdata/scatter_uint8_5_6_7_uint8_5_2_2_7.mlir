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
      stablehlo.return %arg1 : tensor<ui8>
    }) : (tensor<5x6x7xui8>, tensor<2x2x1xi64>, tensor<5x2x2x7xui8>) -> tensor<5x6x7xui8>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<5x6x7xui8>, tensor<5x6x7xui8>) -> ()
    return %2 : tensor<5x6x7xui8>
  }
  func.func private @inputs() -> (tensor<5x6x7xui8> {mhlo.layout_mode = "default"}, tensor<5x2x2x7xui8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x040402020507070202070201040101020002030001020202020000030401000104070500050107050101060200040002000405000402030002000301040201000002010200030201070004000201020401010204010100000201010100020201020003000001020102000001040003030001000502050002030206000001030403020402020506020001000001020201000000000006020501050200000103000100040502010602000105000303030300000400030305000100000206000405010107040003000102000401000600040004"> : tensor<5x6x7xui8>
    %c_0 = stablehlo.constant dense<"0x0103000201030104000003010400000000010303030004020001030102040200000002020206020501050302000006000401010000010105020006030400020301010401060001010101000002000000050001040101020301010104000500000204010202000102010302010106010002030203000101020000010102000000020102040001000003030000"> : tensor<5x2x2x7xui8>
    return %c, %c_0 : tensor<5x6x7xui8>, tensor<5x2x2x7xui8>
  }
  func.func private @expected() -> (tensor<5x6x7xui8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x010300020103010400000301040000000001030303000402000103010401000104070500050107050101020402000000020202060205010503020000060004010100000101050201070004000201020401010204020006030400020301010401060001010101000002000000050001040001000502050002030206000001010102030101010400050000020401020200010201030201010601000200000103000100040502010602020302030001010200000101020000000201020400010000030300000003000102000401000600040004"> : tensor<5x6x7xui8>
    return %c : tensor<5x6x7xui8>
  }
}
