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
      %3 = stablehlo.multiply %arg0, %arg1 : tensor<f16>
      stablehlo.return %3 : tensor<f16>
    }) : (tensor<2x3xf16>, tensor<1x3x1xi64>, tensor<2x1x3xf16>) -> tensor<2x3xf16>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<2x3xf16>, tensor<2x3xf16>) -> ()
    return %2 : tensor<2x3xf16>
  }
  func.func private @inputs() -> (tensor<2x3xf16> {mhlo.layout_mode = "default"}, tensor<2x1x3xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-8.031250e+00, -9.223630e-01, -7.177730e-01], [2.767580e+00, -1.363280e+00, -2.916020e+00]]> : tensor<2x3xf16>
    %cst_0 = stablehlo.constant dense<[[[3.357420e+00, 1.513670e+00, -5.869140e-01]], [[5.527340e-01, 1.699220e+00, -4.015630e+00]]]> : tensor<2x1x3xf16>
    return %cst, %cst_0 : tensor<2x3xf16>, tensor<2x1x3xf16>
  }
  func.func private @expected() -> (tensor<2x3xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-8.031250e+00, -9.223630e-01, 2.140630e+00], [2.767580e+00, -1.363280e+00, 1.099220e+01]]> : tensor<2x3xf16>
    return %cst : tensor<2x3xf16>
  }
}
