// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<2x3xf64> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<2> : tensor<1x3x1xi64>
    %0:2 = call @inputs() : () -> (tensor<2x3xf64>, tensor<2x1x3xf64>)
    %1 = call @expected() : () -> tensor<2x3xf64>
    %2 = "stablehlo.scatter"(%0#0, %c, %0#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 2>}> ({
    ^bb0(%arg0: tensor<f64>, %arg1: tensor<f64>):
      %3 = stablehlo.minimum %arg0, %arg1 : tensor<f64>
      stablehlo.return %3 : tensor<f64>
    }) : (tensor<2x3xf64>, tensor<1x3x1xi64>, tensor<2x1x3xf64>) -> tensor<2x3xf64>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<2x3xf64>, tensor<2x3xf64>) -> ()
    return %2 : tensor<2x3xf64>
  }
  func.func private @inputs() -> (tensor<2x3xf64> {mhlo.layout_mode = "default"}, tensor<2x1x3xf64> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[0.36122310516137945, 2.6986009434872651, 5.4751384889125818], [-0.23399422136522924, -1.2078836128544281, -2.2954949302425307]]> : tensor<2x3xf64>
    %cst_0 = stablehlo.constant dense<[[[-1.4222101237409046, 2.2291616554877445, 1.0076255190643058]], [[5.8557965706290176, -2.4118192541735386, -2.225702241190167]]]> : tensor<2x1x3xf64>
    return %cst, %cst_0 : tensor<2x3xf64>, tensor<2x1x3xf64>
  }
  func.func private @expected() -> (tensor<2x3xf64> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[0.36122310516137945, 2.6986009434872651, -1.4222101237409046], [-0.23399422136522924, -1.2078836128544281, -2.4118192541735386]]> : tensor<2x3xf64>
    return %cst : tensor<2x3xf64>
  }
}
