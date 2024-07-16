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
      %3 = stablehlo.add %arg0, %arg1 : tensor<f16>
      stablehlo.return %3 : tensor<f16>
    }) : (tensor<4x2x3xf16>, tensor<2xi64>, tensor<2xf16>) -> tensor<4x2x3xf16>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<4x2x3xf16>, tensor<4x2x3xf16>) -> ()
    return %2 : tensor<4x2x3xf16>
  }
  func.func private @inputs() -> (tensor<4x2x3xf16> {mhlo.layout_mode = "default"}, tensor<2xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[[-1.169920e+00, 4.914550e-01, 4.372560e-01], [-5.963130e-02, -2.089840e+00, 2.812500e+00]], [[2.240230e+00, -2.693360e+00, 2.982420e+00], [-2.194820e-01, 1.992190e+00, 1.350590e+00]], [[1.885740e+00, -1.938480e+00, 1.156250e+00], [8.725580e-01, -1.982420e+00, 5.625000e+00]], [[1.516600e+00, -3.257810e+00, -2.289060e+00], [7.890630e-01, -1.298830e+00, -1.415040e+00]]]> : tensor<4x2x3xf16>
    %cst_0 = stablehlo.constant dense<[-5.737300e-02, 7.153320e-01]> : tensor<2xf16>
    return %cst, %cst_0 : tensor<4x2x3xf16>, tensor<2xf16>
  }
  func.func private @expected() -> (tensor<4x2x3xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[[-1.169920e+00, 4.914550e-01, 4.372560e-01], [-5.963130e-02, -2.089840e+00, 2.812500e+00]], [[2.240230e+00, -2.693360e+00, 2.982420e+00], [-2.194820e-01, 1.992190e+00, 1.350590e+00]], [[1.885740e+00, -1.938480e+00, 1.156250e+00], [8.725580e-01, -1.982420e+00, 5.625000e+00]], [[1.516600e+00, -3.257810e+00, -2.345700e+00], [7.890630e-01, -1.298830e+00, -6.997070e-01]]]> : tensor<4x2x3xf16>
    return %cst : tensor<4x2x3xf16>
  }
}
