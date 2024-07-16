// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x3xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<4x6xf32>, tensor<4x2xi64>)
    %1 = call @expected() : () -> tensor<4x3xf32>
    %2 = "stablehlo.gather"(%0#0, %0#1) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0, 1], index_vector_dim = 1>, slice_sizes = array<i64: 1, 3>}> : (tensor<4x6xf32>, tensor<4x2xi64>) -> tensor<4x3xf32>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<4x3xf32>, tensor<4x3xf32>) -> ()
    return %2 : tensor<4x3xf32>
  }
  func.func private @inputs() -> (tensor<4x6xf32> {mhlo.layout_mode = "default"}, tensor<4x2xi64> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-4.14172935, 1.45226812, 1.71504426, 0.182875648, 2.24712729, 1.00502801], [3.53438354, 3.140540e+00, -3.78799725, -0.773071169, -7.967960e+00, 2.19334626], [-4.67005682, -0.738040149, 0.920267403, -2.77311182, -2.60643196, -0.117176548], [0.772570908, -3.032180e+00, 1.82749724, -1.43702137, 0.937500596, -4.603724]]> : tensor<4x6xf32>
    %c = stablehlo.constant dense<[[0, 1], [1, 2], [2, 3], [3, 2]]> : tensor<4x2xi64>
    return %cst, %c : tensor<4x6xf32>, tensor<4x2xi64>
  }
  func.func private @expected() -> (tensor<4x3xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[1.45226812, 1.71504426, 0.182875648], [-3.78799725, -0.773071169, -7.967960e+00], [-2.77311182, -2.60643196, -0.117176548], [1.82749724, -1.43702137, 0.937500596]]> : tensor<4x3xf32>
    return %cst : tensor<4x3xf32>
  }
}
