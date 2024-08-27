// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x2x3xbf16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[3, 2]> : tensor<2xi64>
    %0:2 = call @inputs() : () -> (tensor<4x2x3xbf16>, tensor<2xbf16>)
    %1 = call @expected() : () -> tensor<4x2x3xbf16>
    %2 = "stablehlo.scatter"(%0#0, %c, %0#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [0, 2], scatter_dims_to_operand_dims = [0, 2]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<bf16>, %arg1: tensor<bf16>):
      %3 = stablehlo.maximum %arg0, %arg1 : tensor<bf16>
      stablehlo.return %3 : tensor<bf16>
    }) : (tensor<4x2x3xbf16>, tensor<2xi64>, tensor<2xbf16>) -> tensor<4x2x3xbf16>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<4x2x3xbf16>, tensor<4x2x3xbf16>) -> ()
    return %2 : tensor<4x2x3xbf16>
  }
  func.func private @inputs() -> (tensor<4x2x3xbf16> {mhlo.layout_mode = "default"}, tensor<2xbf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[[-9.648430e-01, 1.664060e+00, -9.062500e-01], [5.507810e-01, -2.390630e+00, -7.265630e-01]], [[1.429690e+00, 1.007810e+00, 1.648440e+00], [1.742190e+00, -2.187500e+00, -2.406250e+00]], [[-3.187500e+00, 7.773430e-01, -9.218750e-01], [-2.609380e+00, -1.390630e+00, 3.937500e+00]], [[2.546880e+00, 3.535160e-01, -5.546880e-01], [4.312500e+00, -3.203130e+00, 2.984380e+00]]]> : tensor<4x2x3xbf16>
    %cst_0 = stablehlo.constant dense<[-2.093750e+00, -2.765630e+00]> : tensor<2xbf16>
    return %cst, %cst_0 : tensor<4x2x3xbf16>, tensor<2xbf16>
  }
  func.func private @expected() -> (tensor<4x2x3xbf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[[-9.648430e-01, 1.664060e+00, -9.062500e-01], [5.507810e-01, -2.390630e+00, -7.265630e-01]], [[1.429690e+00, 1.007810e+00, 1.648440e+00], [1.742190e+00, -2.187500e+00, -2.406250e+00]], [[-3.187500e+00, 7.773430e-01, -9.218750e-01], [-2.609380e+00, -1.390630e+00, 3.937500e+00]], [[2.546880e+00, 3.535160e-01, -5.546880e-01], [4.312500e+00, -3.203130e+00, 2.984380e+00]]]> : tensor<4x2x3xbf16>
    return %cst : tensor<4x2x3xbf16>
  }
}
