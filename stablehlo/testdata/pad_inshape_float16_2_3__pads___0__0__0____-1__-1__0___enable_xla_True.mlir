// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<2x3xf16>, tensor<f16>)
    %1 = call @expected() : () -> tensor<2x1xf16>
    %2 = stablehlo.pad %0#0, %0#1, low = [0, -1], high = [0, -1], interior = [0, 0] : (tensor<2x3xf16>, tensor<f16>) -> tensor<2x1xf16>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<2x1xf16>, tensor<2x1xf16>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<2x3xf16>, tensor<f16>) {
    %0 = stablehlo.constant dense<[[1.064300e-03, 6.613730e-04, 8.940690e-04], [-1.350640e-04, 8.244510e-04, -7.734290e-04]]> : tensor<2x3xf16>
    %1 = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    return %0, %1 : tensor<2x3xf16>, tensor<f16>
  }
  func.func private @expected() -> tensor<2x1xf16> {
    %0 = stablehlo.constant dense<[[6.613730e-04], [8.244510e-04]]> : tensor<2x1xf16>
    return %0 : tensor<2x1xf16>
  }
}
