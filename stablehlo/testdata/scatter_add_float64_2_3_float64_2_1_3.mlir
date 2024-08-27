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
      %3 = stablehlo.add %arg0, %arg1 : tensor<f64>
      stablehlo.return %3 : tensor<f64>
    }) : (tensor<2x3xf64>, tensor<1x3x1xi64>, tensor<2x1x3xf64>) -> tensor<2x3xf64>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<2x3xf64>, tensor<2x3xf64>) -> ()
    return %2 : tensor<2x3xf64>
  }
  func.func private @inputs() -> (tensor<2x3xf64> {mhlo.layout_mode = "default"}, tensor<2x1x3xf64> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[0.24353330853720112, 0.35517106245016067, -2.7399508779164528], [-1.0711405401483314, -0.030541696303181529, 1.7652672325670165]]> : tensor<2x3xf64>
    %cst_0 = stablehlo.constant dense<[[[-3.2524828317800165, -3.1230734143458356, -0.14138471376141706]], [[1.1507709935500414, -0.39732819332281605, 4.9637874028875082]]]> : tensor<2x1x3xf64>
    return %cst, %cst_0 : tensor<2x3xf64>, tensor<2x1x3xf64>
  }
  func.func private @expected() -> (tensor<2x3xf64> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[0.24353330853720112, 0.35517106245016067, -9.256891837803721], [-1.0711405401483314, -0.030541696303181529, 7.4824974356817506]]> : tensor<2x3xf64>
    return %cst : tensor<2x3xf64>
  }
}
