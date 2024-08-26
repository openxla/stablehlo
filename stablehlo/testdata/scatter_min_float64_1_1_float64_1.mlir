// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<1x1xf64> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<0> : tensor<1xi64>
    %0:2 = call @inputs() : () -> (tensor<1x1xf64>, tensor<1xf64>)
    %1 = call @expected() : () -> tensor<1x1xf64>
    %2 = "stablehlo.scatter"(%0#0, %c, %0#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<f64>, %arg1: tensor<f64>):
      %3 = stablehlo.minimum %arg0, %arg1 : tensor<f64>
      stablehlo.return %3 : tensor<f64>
    }) : (tensor<1x1xf64>, tensor<1xi64>, tensor<1xf64>) -> tensor<1x1xf64>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<1x1xf64>, tensor<1x1xf64>) -> ()
    return %2 : tensor<1x1xf64>
  }
  func.func private @inputs() -> (tensor<1x1xf64> {mhlo.layout_mode = "default"}, tensor<1xf64> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<2.3147561384066981> : tensor<1x1xf64>
    %cst_0 = stablehlo.constant dense<2.826880031561732> : tensor<1xf64>
    return %cst, %cst_0 : tensor<1x1xf64>, tensor<1xf64>
  }
  func.func private @expected() -> (tensor<1x1xf64> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<2.3147561384066981> : tensor<1x1xf64>
    return %cst : tensor<1x1xf64>
  }
}
