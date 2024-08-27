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
      stablehlo.return %arg1 : tensor<f16>
    }) : (tensor<4x2x3xf16>, tensor<2xi64>, tensor<2xf16>) -> tensor<4x2x3xf16>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<4x2x3xf16>, tensor<4x2x3xf16>) -> ()
    return %2 : tensor<4x2x3xf16>
  }
  func.func private @inputs() -> (tensor<4x2x3xf16> {mhlo.layout_mode = "default"}, tensor<2xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[[6.870120e-01, 2.033200e+00, 2.789060e+00], [1.398440e+00, 7.309570e-01, 2.519530e+00]], [[-3.044920e+00, 2.149660e-01, -5.921880e+00], [-5.277340e+00, 1.605470e+00, 4.816410e+00]], [[-2.134770e+00, -4.492190e-01, -1.436520e+00], [-1.509770e+00, 1.167970e+00, -3.921880e+00]], [[9.851560e+00, 1.700200e+00, -1.511720e+00], [-2.066410e+00, 3.126950e+00, 2.890630e+00]]]> : tensor<4x2x3xf16>
    %cst_0 = stablehlo.constant dense<[-5.906250e+00, 2.890630e+00]> : tensor<2xf16>
    return %cst, %cst_0 : tensor<4x2x3xf16>, tensor<2xf16>
  }
  func.func private @expected() -> (tensor<4x2x3xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[[6.870120e-01, 2.033200e+00, 2.789060e+00], [1.398440e+00, 7.309570e-01, 2.519530e+00]], [[-3.044920e+00, 2.149660e-01, -5.921880e+00], [-5.277340e+00, 1.605470e+00, 4.816410e+00]], [[-2.134770e+00, -4.492190e-01, -1.436520e+00], [-1.509770e+00, 1.167970e+00, -3.921880e+00]], [[9.851560e+00, 1.700200e+00, -5.906250e+00], [-2.066410e+00, 3.126950e+00, 2.890630e+00]]]> : tensor<4x2x3xf16>
    return %cst : tensor<4x2x3xf16>
  }
}
