// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3x5x4xbf16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<1> : tensor<2x1xi64>
    %0:2 = call @inputs() : () -> (tensor<3x5x4xbf16>, tensor<3x2x4xbf16>)
    %1 = call @expected() : () -> tensor<3x5x4xbf16>
    %2 = "stablehlo.scatter"(%0#0, %c, %0#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 2], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>}> ({
    ^bb0(%arg0: tensor<bf16>, %arg1: tensor<bf16>):
      %3 = stablehlo.multiply %arg0, %arg1 : tensor<bf16>
      stablehlo.return %3 : tensor<bf16>
    }) : (tensor<3x5x4xbf16>, tensor<2x1xi64>, tensor<3x2x4xbf16>) -> tensor<3x5x4xbf16>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<3x5x4xbf16>, tensor<3x5x4xbf16>) -> ()
    return %2 : tensor<3x5x4xbf16>
  }
  func.func private @inputs() -> (tensor<3x5x4xbf16> {mhlo.layout_mode = "default"}, tensor<3x2x4xbf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[[2.656250e+00, 4.875000e+00, -1.234380e+00, -2.125000e+00], [-2.687500e+00, -2.226560e-01, -2.203130e+00, -1.101560e+00], [3.890630e+00, 1.092530e-02, -2.234380e+00, 2.687500e+00], [-1.923830e-01, 3.093750e+00, 2.234380e+00, -2.890630e-01], [-8.496090e-02, -4.781250e+00, -2.187500e+00, -3.437500e+00]], [[-2.156250e+00, 3.750000e+00, 6.781250e+00, 4.843750e+00], [-3.703130e+00, 1.898440e+00, -6.531250e+00, -2.781250e+00], [3.671880e+00, 1.578130e+00, -1.718750e+00, 1.078130e+00], [3.906250e-01, -3.359380e+00, 7.656250e-01, 2.875000e+00], [-4.980470e-01, -8.007810e-01, 9.414060e-01, 2.796880e+00]], [[-4.406250e+00, 8.740230e-02, -4.125000e+00, -2.671880e+00], [1.664060e+00, 3.140630e+00, -2.765630e+00, -5.562500e+00], [-2.656250e+00, -8.593750e-01, -1.476560e+00, 4.492190e-01], [5.375000e+00, 1.507810e+00, -3.500000e+00, 1.898440e+00], [-2.328130e+00, 2.490230e-01, -9.101560e-01, -9.921870e-01]]]> : tensor<3x5x4xbf16>
    %cst_0 = stablehlo.constant dense<[[[-3.164060e-01, -3.750000e+00, 8.515620e-01, 3.843750e+00], [-2.031250e+00, 6.015630e-01, -1.531250e+00, -8.544920e-02]], [[4.718750e+00, -4.468750e+00, -9.062500e-01, 4.453130e-01], [6.500000e+00, -3.964840e-01, -8.632810e-01, 2.234380e+00]], [[3.843750e+00, -8.875000e+00, -1.757810e+00, -6.406250e+00], [-1.984380e+00, -7.304680e-01, -4.218750e-01, -3.453130e+00]]]> : tensor<3x2x4xbf16>
    return %cst, %cst_0 : tensor<3x5x4xbf16>, tensor<3x2x4xbf16>
  }
  func.func private @expected() -> (tensor<3x5x4xbf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[[2.656250e+00, 4.875000e+00, -1.234380e+00, -2.125000e+00], [-1.726560e+00, 5.039060e-01, 2.875000e+00, 3.613280e-01], [3.890630e+00, 1.092530e-02, -2.234380e+00, 2.687500e+00], [-1.923830e-01, 3.093750e+00, 2.234380e+00, -2.890630e-01], [-8.496090e-02, -4.781250e+00, -2.187500e+00, -3.437500e+00]], [[-2.156250e+00, 3.750000e+00, 6.781250e+00, 4.843750e+00], [-1.140000e+02, 3.375000e+00, -5.093750e+00, -2.781250e+00], [3.671880e+00, 1.578130e+00, -1.718750e+00, 1.078130e+00], [3.906250e-01, -3.359380e+00, 7.656250e-01, 2.875000e+00], [-4.980470e-01, -8.007810e-01, 9.414060e-01, 2.796880e+00]], [[-4.406250e+00, 8.740230e-02, -4.125000e+00, -2.671880e+00], [-1.268750e+01, 2.037500e+01, -2.062500e+00, -1.235000e+02], [-2.656250e+00, -8.593750e-01, -1.476560e+00, 4.492190e-01], [5.375000e+00, 1.507810e+00, -3.500000e+00, 1.898440e+00], [-2.328130e+00, 2.490230e-01, -9.101560e-01, -9.921870e-01]]]> : tensor<3x5x4xbf16>
    return %cst : tensor<3x5x4xbf16>
  }
}
