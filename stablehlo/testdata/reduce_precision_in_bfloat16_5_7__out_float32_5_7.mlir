// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<5x7xbf16>
    %1 = call @expected() : () -> tensor<5x7xbf16>
    %2 = stablehlo.reduce_precision %0, format = e8m23 : tensor<5x7xbf16>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<5x7xbf16>, tensor<5x7xbf16>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<5x7xbf16> {
    %0 = stablehlo.constant dense<[[1.640630e+00, -4.370120e-02, 3.843750e+00, -1.992190e+00, -3.265630e+00, -1.960940e+00, -4.906250e+00], [-3.085940e-01, 1.914060e+00, 6.562500e-01, 3.027340e-02, 2.703130e+00, -4.687500e-01, 1.640630e+00], [-3.140630e+00, -4.437500e+00, 2.781250e+00, -2.062500e+00, -5.781250e+00, 7.910150e-02, -4.511720e-01], [-2.906250e+00, -2.812500e+00, -3.000000e+00, 2.296880e+00, 3.250000e+00, 1.656250e+00, 9.726560e-01], [-6.640630e-01, 1.953130e+00, -2.437500e+00, 8.554680e-01, -2.093750e+00, 1.125000e+00, -8.046880e-01]]> : tensor<5x7xbf16>
    return %0 : tensor<5x7xbf16>
  }
  func.func private @expected() -> tensor<5x7xbf16> {
    %0 = stablehlo.constant dense<[[1.640630e+00, -4.370120e-02, 3.843750e+00, -1.992190e+00, -3.265630e+00, -1.960940e+00, -4.906250e+00], [-3.085940e-01, 1.914060e+00, 6.562500e-01, 3.027340e-02, 2.703130e+00, -4.687500e-01, 1.640630e+00], [-3.140630e+00, -4.437500e+00, 2.781250e+00, -2.062500e+00, -5.781250e+00, 7.910150e-02, -4.511720e-01], [-2.906250e+00, -2.812500e+00, -3.000000e+00, 2.296880e+00, 3.250000e+00, 1.656250e+00, 9.726560e-01], [-6.640630e-01, 1.953130e+00, -2.437500e+00, 8.554680e-01, -2.093750e+00, 1.125000e+00, -8.046880e-01]]> : tensor<5x7xbf16>
    return %0 : tensor<5x7xbf16>
  }
}
