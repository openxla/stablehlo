// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<5x7xbf16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<5x7xbf16>
    %1 = call @expected() : () -> tensor<5x7xbf16>
    %2 = stablehlo.reduce_precision %0, format = e11m52 : tensor<5x7xbf16>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<5x7xbf16>, tensor<5x7xbf16>) -> ()
    return %2 : tensor<5x7xbf16>
  }
  func.func private @inputs() -> (tensor<5x7xbf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-3.406250e+00, 3.093750e+00, -5.375000e+00, -2.734380e+00, 1.191410e-01, 2.375000e+00, 7.531250e+00], [-6.875000e+00, 3.000000e+00, -3.015630e+00, -1.828130e+00, -7.187500e-01, -2.624510e-02, 3.007810e-01], [-4.687500e+00, 1.953130e+00, 1.046880e+00, 1.671880e+00, -4.277340e-01, -3.062500e+00, -6.250000e+00], [3.750000e+00, -2.000000e+00, 1.726560e+00, -5.625000e+00, 1.289060e+00, -4.531250e+00, 2.984380e+00], [3.242190e-01, -1.218750e+00, 2.575680e-02, -3.222660e-01, 6.562500e+00, 3.312500e+00, 1.945310e+00]]> : tensor<5x7xbf16>
    return %cst : tensor<5x7xbf16>
  }
  func.func private @expected() -> (tensor<5x7xbf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-3.406250e+00, 3.093750e+00, -5.375000e+00, -2.734380e+00, 1.191410e-01, 2.375000e+00, 7.531250e+00], [-6.875000e+00, 3.000000e+00, -3.015630e+00, -1.828130e+00, -7.187500e-01, -2.624510e-02, 3.007810e-01], [-4.687500e+00, 1.953130e+00, 1.046880e+00, 1.671880e+00, -4.277340e-01, -3.062500e+00, -6.250000e+00], [3.750000e+00, -2.000000e+00, 1.726560e+00, -5.625000e+00, 1.289060e+00, -4.531250e+00, 2.984380e+00], [3.242190e-01, -1.218750e+00, 2.575680e-02, -3.222660e-01, 6.562500e+00, 3.312500e+00, 1.945310e+00]]> : tensor<5x7xbf16>
    return %cst : tensor<5x7xbf16>
  }
}
