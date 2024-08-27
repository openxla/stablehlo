// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<2x3x2xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<2x3x3xf32>, tensor<2x3xi64>)
    %1 = call @expected() : () -> tensor<2x3x2xf32>
    %2 = "stablehlo.gather"(%0#0, %0#1) <{dimension_numbers = #stablehlo.gather<offset_dims = [1, 2], collapsed_slice_dims = [0], start_index_map = [0, 1, 2], index_vector_dim = 1>, slice_sizes = array<i64: 1, 3, 2>}> : (tensor<2x3x3xf32>, tensor<2x3xi64>) -> tensor<2x3x2xf32>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<2x3x2xf32>, tensor<2x3x2xf32>) -> ()
    return %2 : tensor<2x3x2xf32>
  }
  func.func private @inputs() -> (tensor<2x3x3xf32> {mhlo.layout_mode = "default"}, tensor<2x3xi64> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[[0.0618097708, 0.232754916, 0.668427229], [4.27152205, -3.72517085, -3.77432418], [1.83894026, -4.33732605, -1.31484962]], [[-3.41498137, 0.911645174, 0.27581653], [-1.31400931, 0.393603295, 2.46858478], [-7.053160e-01, 2.62368941, 3.240237]]]> : tensor<2x3x3xf32>
    %c = stablehlo.constant dense<[[0, 1, 0], [1, 2, 1]]> : tensor<2x3xi64>
    return %cst, %c : tensor<2x3x3xf32>, tensor<2x3xi64>
  }
  func.func private @expected() -> (tensor<2x3x2xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[[0.0618097708, 0.232754916], [4.27152205, -3.72517085], [1.83894026, -4.33732605]], [[0.911645174, 0.27581653], [0.393603295, 2.46858478], [2.62368941, 3.240237]]]> : tensor<2x3x2xf32>
    return %cst : tensor<2x3x2xf32>
  }
}
