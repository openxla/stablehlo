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
      %3 = stablehlo.multiply %arg0, %arg1 : tensor<bf16>
      stablehlo.return %3 : tensor<bf16>
    }) : (tensor<4x2x3xbf16>, tensor<2xi64>, tensor<2xbf16>) -> tensor<4x2x3xbf16>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<4x2x3xbf16>, tensor<4x2x3xbf16>) -> ()
    return %2 : tensor<4x2x3xbf16>
  }
  func.func private @inputs() -> (tensor<4x2x3xbf16> {mhlo.layout_mode = "default"}, tensor<2xbf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[[4.343750e+00, -4.687500e+00, -3.406250e+00], [5.117190e-01, -2.796880e+00, 3.554690e-01]], [[4.625000e+00, 1.375000e+00, 2.328130e+00], [-1.976560e+00, -3.500000e+00, -3.750000e+00]], [[-4.906250e+00, -1.914060e+00, -2.250000e+00], [-5.625000e+00, -3.578130e+00, 1.476560e+00]], [[2.312500e+00, -4.003910e-01, -2.437500e+00], [-2.687500e+00, -1.906250e+00, -4.343750e+00]]]> : tensor<4x2x3xbf16>
    %cst_0 = stablehlo.constant dense<[9.023430e-01, -4.125000e+00]> : tensor<2xbf16>
    return %cst, %cst_0 : tensor<4x2x3xbf16>, tensor<2xbf16>
  }
  func.func private @expected() -> (tensor<4x2x3xbf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[[4.343750e+00, -4.687500e+00, -3.406250e+00], [5.117190e-01, -2.796880e+00, 3.554690e-01]], [[4.625000e+00, 1.375000e+00, 2.328130e+00], [-1.976560e+00, -3.500000e+00, -3.750000e+00]], [[-4.906250e+00, -1.914060e+00, -2.250000e+00], [-5.625000e+00, -3.578130e+00, 1.476560e+00]], [[2.312500e+00, -4.003910e-01, -2.203130e+00], [-2.687500e+00, -1.906250e+00, 1.787500e+01]]]> : tensor<4x2x3xbf16>
    return %cst : tensor<4x2x3xbf16>
  }
}
