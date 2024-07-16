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
      stablehlo.return %arg1 : tensor<bf16>
    }) : (tensor<4x2x3xbf16>, tensor<2xi64>, tensor<2xbf16>) -> tensor<4x2x3xbf16>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<4x2x3xbf16>, tensor<4x2x3xbf16>) -> ()
    return %2 : tensor<4x2x3xbf16>
  }
  func.func private @inputs() -> (tensor<4x2x3xbf16> {mhlo.layout_mode = "default"}, tensor<2xbf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[[-7.421880e-01, -3.265630e+00, -5.968750e+00], [-2.171880e+00, 1.976560e+00, -5.507810e-01]], [[-8.203130e-01, 2.093750e+00, 4.406250e+00], [-1.726560e+00, 2.109380e+00, 3.812500e+00]], [[-2.812500e+00, 4.746090e-01, 3.625000e+00], [-6.750000e+00, 3.453130e+00, -8.476560e-01]], [[1.132810e+00, 5.156250e+00, 5.078130e-02], [4.500000e+00, 2.187500e+00, -2.156250e+00]]]> : tensor<4x2x3xbf16>
    %cst_0 = stablehlo.constant dense<[5.718750e+00, 5.664060e-01]> : tensor<2xbf16>
    return %cst, %cst_0 : tensor<4x2x3xbf16>, tensor<2xbf16>
  }
  func.func private @expected() -> (tensor<4x2x3xbf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[[-7.421880e-01, -3.265630e+00, -5.968750e+00], [-2.171880e+00, 1.976560e+00, -5.507810e-01]], [[-8.203130e-01, 2.093750e+00, 4.406250e+00], [-1.726560e+00, 2.109380e+00, 3.812500e+00]], [[-2.812500e+00, 4.746090e-01, 3.625000e+00], [-6.750000e+00, 3.453130e+00, -8.476560e-01]], [[1.132810e+00, 5.156250e+00, 5.718750e+00], [4.500000e+00, 2.187500e+00, 5.664060e-01]]]> : tensor<4x2x3xbf16>
    return %cst : tensor<4x2x3xbf16>
  }
}
