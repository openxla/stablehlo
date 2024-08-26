// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<5x7xf16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<5x7xf16>
    %1 = call @expected() : () -> tensor<5x7xf16>
    %2 = stablehlo.reduce_precision %0, format = e11m52 : tensor<5x7xf16>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<5x7xf16>, tensor<5x7xf16>) -> ()
    return %2 : tensor<5x7xf16>
  }
  func.func private @inputs() -> (tensor<5x7xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-1.533200e+00, -5.031250e+00, -4.355470e+00, -1.416020e+00, -1.691410e+00, -9.980460e-01, -1.057620e+00], [6.386710e+00, 9.755850e-01, 1.442380e+00, 3.042970e+00, -3.447270e+00, -2.392580e-01, 3.384770e+00], [1.333010e-01, -2.309570e-01, 1.952150e+00, 4.511720e+00, 2.937500e+00, 2.779300e+00, -7.647710e-02], [4.785160e+00, -2.441410e+00, 2.738280e+00, 4.046880e+00, 3.496090e+00, 4.386720e+00, -3.058590e+00], [9.848630e-01, -1.439450e+00, 1.610350e+00, 5.605460e-01, -6.552730e-01, -4.820310e+00, 3.820800e-02]]> : tensor<5x7xf16>
    return %cst : tensor<5x7xf16>
  }
  func.func private @expected() -> (tensor<5x7xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-1.533200e+00, -5.031250e+00, -4.355470e+00, -1.416020e+00, -1.691410e+00, -9.980460e-01, -1.057620e+00], [6.386710e+00, 9.755850e-01, 1.442380e+00, 3.042970e+00, -3.447270e+00, -2.392580e-01, 3.384770e+00], [1.333010e-01, -2.309570e-01, 1.952150e+00, 4.511720e+00, 2.937500e+00, 2.779300e+00, -7.647710e-02], [4.785160e+00, -2.441410e+00, 2.738280e+00, 4.046880e+00, 3.496090e+00, 4.386720e+00, -3.058590e+00], [9.848630e-01, -1.439450e+00, 1.610350e+00, 5.605460e-01, -6.552730e-01, -4.820310e+00, 3.820800e-02]]> : tensor<5x7xf16>
    return %cst : tensor<5x7xf16>
  }
}
