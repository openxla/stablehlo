// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<2x3xf16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<2> : tensor<1x3x1xi64>
    %0:2 = call @inputs() : () -> (tensor<2x3xf16>, tensor<2x1x3xf16>)
    %1 = call @expected() : () -> tensor<2x3xf16>
    %2 = "stablehlo.scatter"(%0#0, %c, %0#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 2>}> ({
    ^bb0(%arg0: tensor<f16>, %arg1: tensor<f16>):
      %3 = stablehlo.minimum %arg0, %arg1 : tensor<f16>
      stablehlo.return %3 : tensor<f16>
    }) : (tensor<2x3xf16>, tensor<1x3x1xi64>, tensor<2x1x3xf16>) -> tensor<2x3xf16>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<2x3xf16>, tensor<2x3xf16>) -> ()
    return %2 : tensor<2x3xf16>
  }
  func.func private @inputs() -> (tensor<2x3xf16> {mhlo.layout_mode = "default"}, tensor<2x1x3xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-1.833980e+00, -1.094730e+00, 2.414060e+00], [1.220700e+00, -2.996090e+00, 3.056640e+00]]> : tensor<2x3xf16>
    %cst_0 = stablehlo.constant dense<[[[-8.945310e-01, -2.839360e-01, 3.564450e+00]], [[4.648440e+00, -2.574220e+00, 2.992190e+00]]]> : tensor<2x1x3xf16>
    return %cst, %cst_0 : tensor<2x3xf16>, tensor<2x1x3xf16>
  }
  func.func private @expected() -> (tensor<2x3xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-1.833980e+00, -1.094730e+00, -8.945310e-01], [1.220700e+00, -2.996090e+00, -2.574220e+00]]> : tensor<2x3xf16>
    return %cst : tensor<2x3xf16>
  }
}
