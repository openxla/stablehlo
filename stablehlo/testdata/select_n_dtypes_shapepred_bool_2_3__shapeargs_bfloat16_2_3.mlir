// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:3 = call @inputs() : () -> (tensor<2x3xi1>, tensor<2x3xbf16>, tensor<2x3xbf16>)
    %1 = call @expected() : () -> tensor<2x3xbf16>
    %2 = stablehlo.select %0#0, %0#2, %0#1 : tensor<2x3xi1>, tensor<2x3xbf16>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<2x3xbf16>, tensor<2x3xbf16>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<2x3xi1>, tensor<2x3xbf16>, tensor<2x3xbf16>) {
    %0 = stablehlo.constant dense<true> : tensor<2x3xi1>
    %1 = stablehlo.constant dense<[[-2.406250e+00, 3.691410e-01, 6.406250e-01], [-5.562500e+00, -1.250000e+00, -4.000000e+00]]> : tensor<2x3xbf16>
    %2 = stablehlo.constant dense<[[-2.531250e+00, 1.132810e+00, 3.093750e+00], [1.375000e+00, -2.218750e+00, -5.625000e-01]]> : tensor<2x3xbf16>
    return %0, %1, %2 : tensor<2x3xi1>, tensor<2x3xbf16>, tensor<2x3xbf16>
  }
  func.func private @expected() -> tensor<2x3xbf16> {
    %0 = stablehlo.constant dense<[[-2.531250e+00, 1.132810e+00, 3.093750e+00], [1.375000e+00, -2.218750e+00, -5.625000e-01]]> : tensor<2x3xbf16>
    return %0 : tensor<2x3xbf16>
  }
}
