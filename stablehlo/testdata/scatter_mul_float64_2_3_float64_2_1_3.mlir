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
      %3 = stablehlo.multiply %arg0, %arg1 : tensor<f64>
      stablehlo.return %3 : tensor<f64>
    }) : (tensor<2x3xf64>, tensor<1x3x1xi64>, tensor<2x1x3xf64>) -> tensor<2x3xf64>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<2x3xf64>, tensor<2x3xf64>) -> ()
    return %2 : tensor<2x3xf64>
  }
  func.func private @inputs() -> (tensor<2x3xf64> {mhlo.layout_mode = "default"}, tensor<2x1x3xf64> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-0.3356875496195324, -0.30514671473155519, 1.9629755390462527], [-1.9110390206632146, 3.0250865442748971, 0.047292113838280039]]> : tensor<2x3xf64>
    %cst_0 = stablehlo.constant dense<[[[0.2595826160226582, -4.2632347015298642, 3.2694717235449002]], [[2.6332022840408809, 0.46558743510622791, 1.0209954953443161]]]> : tensor<2x1x3xf64>
    return %cst, %cst_0 : tensor<2x3xf64>, tensor<2x1x3xf64>
  }
  func.func private @expected() -> (tensor<2x3xf64> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-0.3356875496195324, -0.30514671473155519, -7.1024358631119373], [-1.9110390206632146, 3.0250865442748971, 0.059196772210423576]]> : tensor<2x3xf64>
    return %cst : tensor<2x3xf64>
  }
}
