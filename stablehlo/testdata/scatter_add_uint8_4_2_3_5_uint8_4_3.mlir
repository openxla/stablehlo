// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x2x3x5xui8> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[0, 4]> : tensor<2xi64>
    %0:2 = call @inputs() : () -> (tensor<4x2x3x5xui8>, tensor<4x3xui8>)
    %1 = call @expected() : () -> tensor<4x2x3x5xui8>
    %2 = "stablehlo.scatter"(%0#0, %c, %0#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1, 3], scatter_dims_to_operand_dims = [1, 3]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<ui8>, %arg1: tensor<ui8>):
      %3 = stablehlo.add %arg0, %arg1 : tensor<ui8>
      stablehlo.return %3 : tensor<ui8>
    }) : (tensor<4x2x3x5xui8>, tensor<2xi64>, tensor<4x3xui8>) -> tensor<4x2x3x5xui8>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<4x2x3x5xui8>, tensor<4x2x3x5xui8>) -> ()
    return %2 : tensor<4x2x3x5xui8>
  }
  func.func private @inputs() -> (tensor<4x2x3x5xui8> {mhlo.layout_mode = "default"}, tensor<4x3xui8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x000201040103050002020401000200000200020402010200010000030002000300030403010101000100000000010302050101020201000000050304010603010000010101010203040201050204030202050000020303000003020100050504030002020202010101020202090002000001000201000402"> : tensor<4x2x3x5xui8>
    %c_0 = stablehlo.constant dense<[[0, 1, 0], [1, 2, 2], [0, 3, 2], [1, 1, 1]]> : tensor<4x3xui8>
    return %c, %c_0 : tensor<4x2x3x5xui8>, tensor<4x3xui8>
  }
  func.func private @expected() -> (tensor<4x2x3x5xui8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x000201040103050002030401000200000200020402010200010000030002000300030503010101020100000002010302050101020201000000050304010603010000010101040203040203050204030202050000020303000003020100050604030002030202010102020202090002000001000201000402"> : tensor<4x2x3x5xui8>
    return %c : tensor<4x2x3x5xui8>
  }
}
