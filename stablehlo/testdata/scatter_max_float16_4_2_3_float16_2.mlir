// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x2x3xf16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[3, 2]> : tensor<2xi64>
    %0:2 = call @inputs() : () -> (tensor<4x2x3xf16>, tensor<2xf16>)
    %1 = call @expected() : () -> tensor<4x2x3xf16>
    %2 = "stablehlo.scatter"(%0#0, %c, %0#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [0, 2], scatter_dims_to_operand_dims = [0, 2]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<f16>, %arg1: tensor<f16>):
      %3 = stablehlo.maximum %arg0, %arg1 : tensor<f16>
      stablehlo.return %3 : tensor<f16>
    }) : (tensor<4x2x3xf16>, tensor<2xi64>, tensor<2xf16>) -> tensor<4x2x3xf16>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<4x2x3xf16>, tensor<4x2x3xf16>) -> ()
    return %2 : tensor<4x2x3xf16>
  }
  func.func private @inputs() -> (tensor<4x2x3xf16> {mhlo.layout_mode = "default"}, tensor<2xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[[-1.750000e+00, -6.953130e+00, 1.501950e+00], [-9.409170e-01, -2.517580e+00, 1.010740e+00]], [[-2.265630e+00, -4.507810e+00, 8.435050e-02], [2.167970e+00, 2.337650e-01, -8.265630e+00]], [[-5.066410e+00, -2.749020e-01, -1.154300e+00], [-2.201170e+00, 1.986330e+00, 4.780270e-01]], [[1.514890e-01, 6.554690e+00, 2.150390e+00], [4.082030e+00, 1.086910e+00, -5.820310e-01]]]> : tensor<4x2x3xf16>
    %cst_0 = stablehlo.constant dense<[3.802730e+00, -2.054690e+00]> : tensor<2xf16>
    return %cst, %cst_0 : tensor<4x2x3xf16>, tensor<2xf16>
  }
  func.func private @expected() -> (tensor<4x2x3xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[[-1.750000e+00, -6.953130e+00, 1.501950e+00], [-9.409170e-01, -2.517580e+00, 1.010740e+00]], [[-2.265630e+00, -4.507810e+00, 8.435050e-02], [2.167970e+00, 2.337650e-01, -8.265630e+00]], [[-5.066410e+00, -2.749020e-01, -1.154300e+00], [-2.201170e+00, 1.986330e+00, 4.780270e-01]], [[1.514890e-01, 6.554690e+00, 3.802730e+00], [4.082030e+00, 1.086910e+00, -5.820310e-01]]]> : tensor<4x2x3xf16>
    return %cst : tensor<4x2x3xf16>
  }
}
