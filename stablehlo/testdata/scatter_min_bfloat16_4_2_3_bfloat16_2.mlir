// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x2x3xbf16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[3, 2]> : tensor<2xi64>
    %0:2 = call @inputs() : () -> (tensor<4x2x3xbf16>, tensor<2xbf16>)
    %1 = call @expected() : () -> tensor<4x2x3xbf16>
    %2 = "stablehlo.scatter"(%0#0, %c, %0#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [0, 2], scatter_dims_to_operand_dims = [0, 2]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<bf16>, %arg1: tensor<bf16>):
      %3 = stablehlo.minimum %arg0, %arg1 : tensor<bf16>
      stablehlo.return %3 : tensor<bf16>
    }) : (tensor<4x2x3xbf16>, tensor<2xi64>, tensor<2xbf16>) -> tensor<4x2x3xbf16>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<4x2x3xbf16>, tensor<4x2x3xbf16>) -> ()
    return %2 : tensor<4x2x3xbf16>
  }
  func.func private @inputs() -> (tensor<4x2x3xbf16> {mhlo.layout_mode = "default"}, tensor<2xbf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[[-4.218750e+00, -1.023440e+00, 6.542960e-02], [2.046880e+00, -2.773440e-01, -5.218750e+00]], [[2.562500e+00, -3.921880e+00, 2.140630e+00], [8.398430e-01, -4.781250e+00, 1.015630e+00]], [[1.007810e+00, 2.265630e+00, -3.500000e+00], [5.500000e+00, 2.234380e+00, 2.500000e+00]], [[2.281250e+00, -2.250000e+00, -1.210940e+00], [2.080080e-01, -1.552730e-01, -2.390630e+00]]]> : tensor<4x2x3xbf16>
    %cst_0 = stablehlo.constant dense<[2.093750e+00, 1.976560e+00]> : tensor<2xbf16>
    return %cst, %cst_0 : tensor<4x2x3xbf16>, tensor<2xbf16>
  }
  func.func private @expected() -> (tensor<4x2x3xbf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[[-4.218750e+00, -1.023440e+00, 6.542960e-02], [2.046880e+00, -2.773440e-01, -5.218750e+00]], [[2.562500e+00, -3.921880e+00, 2.140630e+00], [8.398430e-01, -4.781250e+00, 1.015630e+00]], [[1.007810e+00, 2.265630e+00, -3.500000e+00], [5.500000e+00, 2.234380e+00, 2.500000e+00]], [[2.281250e+00, -2.250000e+00, -1.210940e+00], [2.080080e-01, -1.552730e-01, -2.390630e+00]]]> : tensor<4x2x3xbf16>
    return %cst : tensor<4x2x3xbf16>
  }
}
