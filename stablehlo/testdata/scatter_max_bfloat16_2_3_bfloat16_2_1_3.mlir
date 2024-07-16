// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<2x3xbf16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<2> : tensor<1x3x1xi64>
    %0:2 = call @inputs() : () -> (tensor<2x3xbf16>, tensor<2x1x3xbf16>)
    %1 = call @expected() : () -> tensor<2x3xbf16>
    %2 = "stablehlo.scatter"(%0#0, %c, %0#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 2>}> ({
    ^bb0(%arg0: tensor<bf16>, %arg1: tensor<bf16>):
      %3 = stablehlo.maximum %arg0, %arg1 : tensor<bf16>
      stablehlo.return %3 : tensor<bf16>
    }) : (tensor<2x3xbf16>, tensor<1x3x1xi64>, tensor<2x1x3xbf16>) -> tensor<2x3xbf16>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<2x3xbf16>, tensor<2x3xbf16>) -> ()
    return %2 : tensor<2x3xbf16>
  }
  func.func private @inputs() -> (tensor<2x3xbf16> {mhlo.layout_mode = "default"}, tensor<2x1x3xbf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-2.990720e-02, 2.171880e+00, -1.523440e-01], [-1.078130e+00, 1.281250e+00, -7.890630e-01]]> : tensor<2x3xbf16>
    %cst_0 = stablehlo.constant dense<[[[2.437500e+00, -1.234380e+00, -3.406250e+00]], [[3.421880e+00, 3.265630e+00, -2.781250e+00]]]> : tensor<2x1x3xbf16>
    return %cst, %cst_0 : tensor<2x3xbf16>, tensor<2x1x3xbf16>
  }
  func.func private @expected() -> (tensor<2x3xbf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-2.990720e-02, 2.171880e+00, 2.437500e+00], [-1.078130e+00, 1.281250e+00, 3.421880e+00]]> : tensor<2x3xbf16>
    return %cst : tensor<2x3xbf16>
  }
}
