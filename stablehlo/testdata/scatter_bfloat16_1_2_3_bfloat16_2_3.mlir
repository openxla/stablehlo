// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<1x2x3xbf16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<0> : tensor<1xi64>
    %0:2 = call @inputs() : () -> (tensor<1x2x3xbf16>, tensor<2x3xbf16>)
    %1 = call @expected() : () -> tensor<1x2x3xbf16>
    %2 = "stablehlo.scatter"(%0#0, %c, %0#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<bf16>, %arg1: tensor<bf16>):
      stablehlo.return %arg1 : tensor<bf16>
    }) : (tensor<1x2x3xbf16>, tensor<1xi64>, tensor<2x3xbf16>) -> tensor<1x2x3xbf16>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<1x2x3xbf16>, tensor<1x2x3xbf16>) -> ()
    return %2 : tensor<1x2x3xbf16>
  }
  func.func private @inputs() -> (tensor<1x2x3xbf16> {mhlo.layout_mode = "default"}, tensor<2x3xbf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[[-1.601560e+00, -2.871090e-01, -3.125000e+00], [-3.343750e+00, -9.296870e-01, 3.300780e-01]]]> : tensor<1x2x3xbf16>
    %cst_0 = stablehlo.constant dense<[[3.261720e-01, -1.734380e+00, -4.093750e+00], [-4.125000e+00, 8.437500e-01, -4.941410e-01]]> : tensor<2x3xbf16>
    return %cst, %cst_0 : tensor<1x2x3xbf16>, tensor<2x3xbf16>
  }
  func.func private @expected() -> (tensor<1x2x3xbf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[[3.261720e-01, -1.734380e+00, -4.093750e+00], [-4.125000e+00, 8.437500e-01, -4.941410e-01]]]> : tensor<1x2x3xbf16>
    return %cst : tensor<1x2x3xbf16>
  }
}
