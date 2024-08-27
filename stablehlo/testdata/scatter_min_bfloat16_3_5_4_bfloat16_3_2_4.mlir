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
      %3 = stablehlo.minimum %arg0, %arg1 : tensor<bf16>
      stablehlo.return %3 : tensor<bf16>
    }) : (tensor<3x5x4xbf16>, tensor<2x1xi64>, tensor<3x2x4xbf16>) -> tensor<3x5x4xbf16>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<3x5x4xbf16>, tensor<3x5x4xbf16>) -> ()
    return %2 : tensor<3x5x4xbf16>
  }
  func.func private @inputs() -> (tensor<3x5x4xbf16> {mhlo.layout_mode = "default"}, tensor<3x2x4xbf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[[-4.843750e+00, -4.312500e+00, 1.937500e+00, -1.132810e+00], [-1.789060e+00, -2.218750e+00, 1.039060e+00, 2.328130e+00], [1.890630e+00, 2.078130e+00, 4.223630e-02, 4.312500e+00], [-1.664060e+00, -3.703130e+00, -2.333980e-01, -1.890630e+00], [5.664060e-01, -2.406250e+00, -4.375000e+00, 4.281250e+00]], [[-8.398430e-01, 2.812500e-01, -1.414060e+00, -1.943360e-01], [1.101560e+00, -3.750000e+00, 1.875000e+00, -3.437500e+00], [2.890630e+00, -1.000000e+00, -2.015630e+00, 9.921870e-01], [-6.312500e+00, -5.500000e+00, 6.171880e-01, 3.968750e+00], [-1.078130e+00, -3.750000e+00, -1.365660e-03, -5.664060e-01]], [[2.031250e+00, 3.937500e+00, -1.843750e+00, 4.437500e+00], [1.859380e+00, -2.250000e+00, 7.421880e-01, 4.750000e+00], [-5.218750e+00, -2.167970e-01, 1.046880e+00, 5.625000e+00], [9.335930e-01, -5.812500e+00, 3.437500e-01, -1.281250e+00], [-3.984380e+00, 6.718750e-01, 5.812500e+00, -9.453120e-01]]]> : tensor<3x5x4xbf16>
    %cst_0 = stablehlo.constant dense<[[[-1.726560e+00, 4.093750e+00, -6.484380e-01, 3.066410e-01], [-1.343750e+00, 3.250000e+00, 7.500000e-01, -4.062500e-01]], [[3.609380e+00, -1.484380e-01, 3.000000e+00, -5.156250e-01], [1.460940e+00, -2.937500e+00, 1.748050e-01, -2.765630e+00]], [[-1.386720e-01, -5.859380e-01, 2.015630e+00, -3.812500e+00], [-1.664060e+00, 1.195310e+00, 3.886720e-01, -7.226560e-01]]]> : tensor<3x2x4xbf16>
    return %cst, %cst_0 : tensor<3x5x4xbf16>, tensor<3x2x4xbf16>
  }
  func.func private @expected() -> (tensor<3x5x4xbf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[[-4.843750e+00, -4.312500e+00, 1.937500e+00, -1.132810e+00], [-1.789060e+00, -2.218750e+00, -6.484380e-01, -4.062500e-01], [1.890630e+00, 2.078130e+00, 4.223630e-02, 4.312500e+00], [-1.664060e+00, -3.703130e+00, -2.333980e-01, -1.890630e+00], [5.664060e-01, -2.406250e+00, -4.375000e+00, 4.281250e+00]], [[-8.398430e-01, 2.812500e-01, -1.414060e+00, -1.943360e-01], [1.101560e+00, -3.750000e+00, 1.748050e-01, -3.437500e+00], [2.890630e+00, -1.000000e+00, -2.015630e+00, 9.921870e-01], [-6.312500e+00, -5.500000e+00, 6.171880e-01, 3.968750e+00], [-1.078130e+00, -3.750000e+00, -1.365660e-03, -5.664060e-01]], [[2.031250e+00, 3.937500e+00, -1.843750e+00, 4.437500e+00], [-1.664060e+00, -2.250000e+00, 3.886720e-01, -3.812500e+00], [-5.218750e+00, -2.167970e-01, 1.046880e+00, 5.625000e+00], [9.335930e-01, -5.812500e+00, 3.437500e-01, -1.281250e+00], [-3.984380e+00, 6.718750e-01, 5.812500e+00, -9.453120e-01]]]> : tensor<3x5x4xbf16>
    return %cst : tensor<3x5x4xbf16>
  }
}
