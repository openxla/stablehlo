// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3x5xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<3x10xf32>, tensor<3x2xi64>)
    %1 = call @expected() : () -> tensor<3x5xf32>
    %2 = "stablehlo.gather"(%0#0, %0#1) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0, 1], index_vector_dim = 1>, slice_sizes = array<i64: 1, 5>}> : (tensor<3x10xf32>, tensor<3x2xi64>) -> tensor<3x5xf32>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<3x5xf32>, tensor<3x5xf32>) -> ()
    return %2 : tensor<3x5xf32>
  }
  func.func private @inputs() -> (tensor<3x10xf32> {mhlo.layout_mode = "default"}, tensor<3x2xi64> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-1.4103384, 4.73668623, 0.673182607, -0.61154145, 1.11993873, 0.331038088, 2.017760e+00, 2.24512839, 0.0639716461, 3.36236548], [-0.542979777, -2.56619191, -1.39182234, 1.95056581, 1.0568105, -1.56906319, -4.38390303, 2.2276547, 2.44901681, 3.73211145], [-0.878893256, -6.06845617, -4.15767241, -5.33732271, 2.93643975, -0.195474297, -1.49406374, -0.341617793, -5.95141411, -2.91100192]]> : tensor<3x10xf32>
    %c = stablehlo.constant dense<[[0, 0], [1, 8], [2, 0]]> : tensor<3x2xi64>
    return %cst, %c : tensor<3x10xf32>, tensor<3x2xi64>
  }
  func.func private @expected() -> (tensor<3x5xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-1.4103384, 4.73668623, 0.673182607, -0.61154145, 1.11993873], [-1.56906319, -4.38390303, 2.2276547, 2.44901681, 3.73211145], [-0.878893256, -6.06845617, -4.15767241, -5.33732271, 2.93643975]]> : tensor<3x5xf32>
    return %cst : tensor<3x5xf32>
  }
}
