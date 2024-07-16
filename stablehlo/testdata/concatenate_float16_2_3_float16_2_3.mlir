// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x3xf16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<2x3xf16>, tensor<2x3xf16>)
    %1 = call @expected() : () -> tensor<4x3xf16>
    %2 = stablehlo.concatenate %0#0, %0#1, dim = 0 : (tensor<2x3xf16>, tensor<2x3xf16>) -> tensor<4x3xf16>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<4x3xf16>, tensor<4x3xf16>) -> ()
    return %2 : tensor<4x3xf16>
  }
  func.func private @inputs() -> (tensor<2x3xf16> {mhlo.layout_mode = "default"}, tensor<2x3xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[4.105470e+00, -1.268550e+00, 2.253910e+00], [-2.400390e+00, -5.878900e-01, -2.328130e+00]]> : tensor<2x3xf16>
    %cst_0 = stablehlo.constant dense<[[-1.853520e+00, 1.364260e+00, -4.183590e+00], [9.213860e-01, -3.865230e+00, -6.499020e-01]]> : tensor<2x3xf16>
    return %cst, %cst_0 : tensor<2x3xf16>, tensor<2x3xf16>
  }
  func.func private @expected() -> (tensor<4x3xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[4.105470e+00, -1.268550e+00, 2.253910e+00], [-2.400390e+00, -5.878900e-01, -2.328130e+00], [-1.853520e+00, 1.364260e+00, -4.183590e+00], [9.213860e-01, -3.865230e+00, -6.499020e-01]]> : tensor<4x3xf16>
    return %cst : tensor<4x3xf16>
  }
}
