// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3x5xbf16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<4x6xbf16>
    %1 = call @expected() : () -> tensor<3x5xbf16>
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %2 = "stablehlo.reduce_window"(%0, %cst) <{window_dimensions = array<i64: 2, 2>}> ({
    ^bb0(%arg0: tensor<bf16>, %arg1: tensor<bf16>):
      %3 = stablehlo.maximum %arg0, %arg1 : tensor<bf16>
      stablehlo.return %3 : tensor<bf16>
    }) : (tensor<4x6xbf16>, tensor<bf16>) -> tensor<3x5xbf16>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<3x5xbf16>, tensor<3x5xbf16>) -> ()
    return %2 : tensor<3x5xbf16>
  }
  func.func private @inputs() -> (tensor<4x6xbf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[1.406250e+00, 6.687500e+00, 4.375000e+00, 4.875000e+00, -1.960940e+00, 3.984380e-01], [-2.921880e+00, 1.494140e-01, -5.742190e-01, 3.484380e+00, -2.593750e+00, 1.923830e-01], [1.257810e+00, 3.625000e+00, -4.418950e-02, 3.164060e-01, -3.781250e+00, -2.843750e+00], [-3.500000e+00, 5.781250e-01, -7.773430e-01, 2.281250e+00, 7.937500e+00, -2.812500e-01]]> : tensor<4x6xbf16>
    return %cst : tensor<4x6xbf16>
  }
  func.func private @expected() -> (tensor<3x5xbf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[6.687500e+00, 6.687500e+00, 4.875000e+00, 4.875000e+00, 1.000000e+00], [3.625000e+00, 3.625000e+00, 3.484380e+00, 3.484380e+00, 1.000000e+00], [3.625000e+00, 3.625000e+00, 2.281250e+00, 7.937500e+00, 7.937500e+00]]> : tensor<3x5xbf16>
    return %cst : tensor<3x5xbf16>
  }
}
