// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<2xf16>, tensor<2xf16>)
    %1 = call @expected() : () -> tensor<2xf16>
    %2 = stablehlo.divide %0#0, %0#1 : tensor<2xf16>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<2xf16>, tensor<2xf16>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<2xf16>, tensor<2xf16>) {
    %0 = stablehlo.constant dense<[6.186520e-01, -3.292850e-02]> : tensor<2xf16>
    %1 = stablehlo.constant dense<[-1.421880e+00, 1.963870e+00]> : tensor<2xf16>
    return %0, %1 : tensor<2xf16>, tensor<2xf16>
  }
  func.func private @expected() -> tensor<2xf16> {
    %0 = stablehlo.constant dense<[-4.350590e-01, -1.676940e-02]> : tensor<2xf16>
    return %0 : tensor<2xf16>
  }
}
