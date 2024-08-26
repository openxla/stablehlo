// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<1x50x3xui8> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<32> : tensor<1xi64>
    %0:2 = call @inputs() : () -> (tensor<1x50x3xui8>, tensor<1x3xui8>)
    %1 = call @expected() : () -> tensor<1x50x3xui8>
    %2 = "stablehlo.scatter"(%0#0, %c, %0#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<ui8>, %arg1: tensor<ui8>):
      %3 = stablehlo.add %arg0, %arg1 : tensor<ui8>
      stablehlo.return %3 : tensor<ui8>
    }) : (tensor<1x50x3xui8>, tensor<1xi64>, tensor<1x3xui8>) -> tensor<1x50x3xui8>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<1x50x3xui8>, tensor<1x50x3xui8>) -> ()
    return %2 : tensor<1x50x3xui8>
  }
  func.func private @inputs() -> (tensor<1x50x3xui8> {mhlo.layout_mode = "default"}, tensor<1x3xui8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x020301020101030303000504010302010002040101020201000300040301000402010103040401000101020200010000040003000004030500030002020200010303030700000202010105010100050004010201000101050706020004010404030101020000010003070002010002010103050103010201000500050001020600060704010101000005000400000102000407000104"> : tensor<1x50x3xui8>
    %c_0 = stablehlo.constant dense<[[0, 3, 3]]> : tensor<1x3xui8>
    return %c, %c_0 : tensor<1x50x3xui8>, tensor<1x3xui8>
  }
  func.func private @expected() -> (tensor<1x50x3xui8> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<"0x020301020101030303000504010302010002040101020201000300040301000402010103040401000101020200010000040003000004030500030002020200010303030700000202010105010100050004010201000101050706020004010404030404020000010003070002010002010103050103010201000500050001020600060704010101000005000400000102000407000104"> : tensor<1x50x3xui8>
    return %c : tensor<1x50x3xui8>
  }
}
