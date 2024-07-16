// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<1x2x3xf64> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<1> : tensor<1xi64>
    %0:2 = call @inputs() : () -> (tensor<1x2x3xf64>, tensor<1x3xf64>)
    %1 = call @expected() : () -> tensor<1x2x3xf64>
    %2 = "stablehlo.scatter"(%0#0, %c, %0#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<f64>, %arg1: tensor<f64>):
      stablehlo.return %arg1 : tensor<f64>
    }) : (tensor<1x2x3xf64>, tensor<1xi64>, tensor<1x3xf64>) -> tensor<1x2x3xf64>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<1x2x3xf64>, tensor<1x2x3xf64>) -> ()
    return %2 : tensor<1x2x3xf64>
  }
  func.func private @inputs() -> (tensor<1x2x3xf64> {mhlo.layout_mode = "default"}, tensor<1x3xf64> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[[0.76579004774690629, -3.4947719913577284, -2.4233477292387779], [-4.479516407994419, 2.2589492699799494, 4.1067706287841226]]]> : tensor<1x2x3xf64>
    %cst_0 = stablehlo.constant dense<[[0.35873862080704749, 0.3362588141650133, 2.4025330572554511]]> : tensor<1x3xf64>
    return %cst, %cst_0 : tensor<1x2x3xf64>, tensor<1x3xf64>
  }
  func.func private @expected() -> (tensor<1x2x3xf64> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[[0.76579004774690629, -3.4947719913577284, -2.4233477292387779], [0.35873862080704749, 0.3362588141650133, 2.4025330572554511]]]> : tensor<1x2x3xf64>
    return %cst : tensor<1x2x3xf64>
  }
}
