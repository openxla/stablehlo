// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<1x3x4xbf16>, tensor<1x4x3xbf16>)
    %1 = call @expected() : () -> tensor<1xbf16>
    %2 = "stablehlo.dot_general"(%0#0, %0#1) {dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2, 1], rhs_contracting_dimensions = [1, 2]>} : (tensor<1x3x4xbf16>, tensor<1x4x3xbf16>) -> tensor<1xbf16>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<1xbf16>, tensor<1xbf16>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<1x3x4xbf16>, tensor<1x4x3xbf16>) {
    %0 = stablehlo.constant dense<[[[-1.289060e+00, 1.601560e+00, 2.910160e-01, 9.726560e-01], [-6.835930e-01, -4.593750e+00, 4.781250e+00, -2.546880e+00], [1.171880e+00, 2.531250e+00, -2.343750e+00, -4.281250e+00]]]> : tensor<1x3x4xbf16>
    %1 = stablehlo.constant dense<[[[-3.515630e+00, 6.406250e+00, -3.375000e+00], [3.710940e-01, 4.406250e+00, -3.015630e+00], [5.531250e+00, -3.515630e+00, -3.281250e+00], [-3.062500e+00, 2.765630e+00, -6.250000e-01]]]> : tensor<1x4x3xbf16>
    return %0, %1 : tensor<1x3x4xbf16>, tensor<1x4x3xbf16>
  }
  func.func private @expected() -> tensor<1xbf16> {
    %0 = stablehlo.constant dense<-4.600000e+01> : tensor<1xbf16>
    return %0 : tensor<1xbf16>
  }
}

