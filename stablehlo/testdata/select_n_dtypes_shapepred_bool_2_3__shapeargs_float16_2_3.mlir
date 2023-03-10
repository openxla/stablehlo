// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:3 = call @inputs() : () -> (tensor<2x3xi1>, tensor<2x3xf16>, tensor<2x3xf16>)
    %1 = call @expected() : () -> tensor<2x3xf16>
    %2 = stablehlo.select %0#0, %0#2, %0#1 : tensor<2x3xi1>, tensor<2x3xf16>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<2x3xf16>, tensor<2x3xf16>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<2x3xi1>, tensor<2x3xf16>, tensor<2x3xf16>) {
    %0 = stablehlo.constant dense<true> : tensor<2x3xi1>
    %1 = stablehlo.constant dense<[[4.510500e-02, -2.294920e+00, -8.781250e+00], [4.699710e-01, 4.824220e-01, -5.222660e+00]]> : tensor<2x3xf16>
    %2 = stablehlo.constant dense<[[-1.615230e+00, 1.829100e+00, 3.408200e+00], [-3.449220e+00, 3.109380e+00, 2.808590e+00]]> : tensor<2x3xf16>
    return %0, %1, %2 : tensor<2x3xi1>, tensor<2x3xf16>, tensor<2x3xf16>
  }
  func.func private @expected() -> tensor<2x3xf16> {
    %0 = stablehlo.constant dense<[[-1.615230e+00, 1.829100e+00, 3.408200e+00], [-3.449220e+00, 3.109380e+00, 2.808590e+00]]> : tensor<2x3xf16>
    return %0 : tensor<2x3xf16>
  }
}
