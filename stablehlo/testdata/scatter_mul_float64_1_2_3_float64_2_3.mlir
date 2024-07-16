// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<1x2x3xf64> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<0> : tensor<1xi64>
    %0:2 = call @inputs() : () -> (tensor<1x2x3xf64>, tensor<2x3xf64>)
    %1 = call @expected() : () -> tensor<1x2x3xf64>
    %2 = "stablehlo.scatter"(%0#0, %c, %0#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<f64>, %arg1: tensor<f64>):
      %3 = stablehlo.multiply %arg0, %arg1 : tensor<f64>
      stablehlo.return %3 : tensor<f64>
    }) : (tensor<1x2x3xf64>, tensor<1xi64>, tensor<2x3xf64>) -> tensor<1x2x3xf64>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<1x2x3xf64>, tensor<1x2x3xf64>) -> ()
    return %2 : tensor<1x2x3xf64>
  }
  func.func private @inputs() -> (tensor<1x2x3xf64> {mhlo.layout_mode = "default"}, tensor<2x3xf64> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[[-1.9496638195342539, -5.4588870676814425, 4.1749035380881958], [-4.9247705077788364, -5.1237762282540311, 0.47912346364803982]]]> : tensor<1x2x3xf64>
    %cst_0 = stablehlo.constant dense<[[-2.0849539812706688, -1.8483073442031195, 3.9152889538532589], [-3.135714283477232, 2.1974411834856986, -4.9377501078138355]]> : tensor<2x3xf64>
    return %cst, %cst_0 : tensor<1x2x3xf64>, tensor<2x3xf64>
  }
  func.func private @expected() -> (tensor<1x2x3xf64> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[[4.0649593426773212, 10.089701058371041, 16.345953706079602], [15.442673224089518, -11.259196898930426, -2.3657919342842471]]]> : tensor<1x2x3xf64>
    return %cst : tensor<1x2x3xf64>
  }
}
