// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<2xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<5xf32>, tensor<2x1xi64>)
    %1 = call @expected() : () -> tensor<2xf32>
    %2 = "stablehlo.gather"(%0#0, %0#1) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = array<i64: 1>}> : (tensor<5xf32>, tensor<2x1xi64>) -> tensor<2xf32>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<2xf32>, tensor<2xf32>) -> ()
    return %2 : tensor<2xf32>
  }
  func.func private @inputs() -> (tensor<5xf32> {mhlo.layout_mode = "default"}, tensor<2x1xi64> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[1.27891302, 0.396668941, 2.52560949, 1.63635993, 2.47598505]> : tensor<5xf32>
    %c = stablehlo.constant dense<[[0], [2]]> : tensor<2x1xi64>
    return %cst, %c : tensor<5xf32>, tensor<2x1xi64>
  }
  func.func private @expected() -> (tensor<2xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[1.27891302, 2.52560949]> : tensor<2xf32>
    return %cst : tensor<2xf32>
  }
}
