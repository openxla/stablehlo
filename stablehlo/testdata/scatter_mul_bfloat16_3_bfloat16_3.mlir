// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3xbf16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[1], [0], [1]]> : tensor<3x1xi64>
    %0:2 = call @inputs() : () -> (tensor<3xbf16>, tensor<3xbf16>)
    %1 = call @expected() : () -> tensor<3xbf16>
    %2 = "stablehlo.scatter"(%0#0, %c, %0#1) <{scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>}> ({
    ^bb0(%arg0: tensor<bf16>, %arg1: tensor<bf16>):
      %3 = stablehlo.multiply %arg0, %arg1 : tensor<bf16>
      stablehlo.return %3 : tensor<bf16>
    }) : (tensor<3xbf16>, tensor<3x1xi64>, tensor<3xbf16>) -> tensor<3xbf16>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<3xbf16>, tensor<3xbf16>) -> ()
    return %2 : tensor<3xbf16>
  }
  func.func private @inputs() -> (tensor<3xbf16> {mhlo.layout_mode = "default"}, tensor<3xbf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[-3.750000e-01, -2.656250e+00, -1.906250e+00]> : tensor<3xbf16>
    %cst_0 = stablehlo.constant dense<[-7.851560e-01, 2.765630e+00, 7.734380e-01]> : tensor<3xbf16>
    return %cst, %cst_0 : tensor<3xbf16>, tensor<3xbf16>
  }
  func.func private @expected() -> (tensor<3xbf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[-1.039060e+00, 1.609380e+00, -1.906250e+00]> : tensor<3xbf16>
    return %cst : tensor<3xbf16>
  }
}
