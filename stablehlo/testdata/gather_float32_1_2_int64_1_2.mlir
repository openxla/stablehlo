// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<1xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<1x2xf32>, tensor<1x2xi64>)
    %1 = call @expected() : () -> tensor<1xf32>
    %2 = "stablehlo.gather"(%0#0, %0#1) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>, slice_sizes = array<i64: 1, 1>}> : (tensor<1x2xf32>, tensor<1x2xi64>) -> tensor<1xf32>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<1xf32>, tensor<1xf32>) -> ()
    return %2 : tensor<1xf32>
  }
  func.func private @inputs() -> (tensor<1x2xf32> {mhlo.layout_mode = "default"}, tensor<1x2xi64> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-4.18829679, -0.571287096]]> : tensor<1x2xf32>
    %c = stablehlo.constant dense<[[0, 1]]> : tensor<1x2xi64>
    return %cst, %c : tensor<1x2xf32>, tensor<1x2xi64>
  }
  func.func private @expected() -> (tensor<1xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<-0.571287096> : tensor<1xf32>
    return %cst : tensor<1xf32>
  }
}
