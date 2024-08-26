// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<2x4x3xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<2x4x3xf32>, tensor<2x1x3xf32>)
    %1 = call @expected() : () -> tensor<2x4x3xf32>
    %2 = stablehlo.broadcast_in_dim %0#1, dims = [0, 1, 2] : (tensor<2x1x3xf32>) -> tensor<2x4x3xf32>
    %3 = stablehlo.divide %0#0, %2 : tensor<2x4x3xf32>
    stablehlo.custom_call @check.expect_close(%3, %1) {has_side_effect = true} : (tensor<2x4x3xf32>, tensor<2x4x3xf32>) -> ()
    return %3 : tensor<2x4x3xf32>
  }
  func.func private @inputs() -> (tensor<2x4x3xf32> {mhlo.layout_mode = "default"}, tensor<2x1x3xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[[-0.330817193, -3.21640301, 1.15269339], [5.66107893, 1.98200059, 3.72667956], [1.36263978, 3.81335855, 0.327093095], [-2.62778568, -2.15805292, -1.09849858]], [[-1.18254316, -2.92176151, -2.32717824], [-1.72833169, 4.70706177, 1.16282248], [3.48577952, 2.32715058, 0.348840296], [0.373283356, 3.3044579, -2.34404302]]]> : tensor<2x4x3xf32>
    %cst_0 = stablehlo.constant dense<[[[3.98198795, -2.23650098, -2.21360612]], [[1.605560e+00, -3.02210736, 1.74050951]]]> : tensor<2x1x3xf32>
    return %cst, %cst_0 : tensor<2x4x3xf32>, tensor<2x1x3xf32>
  }
  func.func private @expected() -> (tensor<2x4x3xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[[-8.307840e-02, 1.43814063, -0.520731032], [1.42167151, -0.88620603, -1.68353331], [0.342200875, -1.70505559, -0.147764817], [-6.599180e-01, 0.964923739, 0.496248454]], [[-0.736530066, 0.96679604, -1.33706725], [-1.07646656, -1.55754292, 0.668093144], [2.17106771, -0.7700423, 0.200424239], [0.23249419, -1.09342837, -1.34675682]]]> : tensor<2x4x3xf32>
    return %cst : tensor<2x4x3xf32>
  }
}
