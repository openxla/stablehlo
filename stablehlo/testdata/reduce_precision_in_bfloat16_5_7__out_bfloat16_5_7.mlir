// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<5x7xbf16>
    %1 = call @expected() : () -> tensor<5x7xbf16>
    %2 = stablehlo.reduce_precision %0, format = e8m7 : tensor<5x7xbf16>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<5x7xbf16>, tensor<5x7xbf16>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<5x7xbf16> {
    %0 = stablehlo.constant dense<[[-3.468750e+00, -1.960940e+00, 1.414060e+00, 1.351560e+00, 2.234380e+00, -1.726560e+00, -2.703130e+00], [-1.296880e+00, 1.281250e+00, 3.078130e+00, 6.171880e-01, 6.601560e-01, 3.718750e+00, 1.562500e+00], [-5.531250e+00, 3.734380e+00, 2.460940e-01, 1.867190e+00, -3.496090e-01, 4.062500e+00, 2.984380e+00], [1.570310e+00, 1.960940e+00, 9.531250e-01, -2.234380e+00, 2.609380e+00, -1.476560e+00, 4.433590e-01], [3.906250e+00, -6.591800e-02, -7.265630e-01, 3.437500e-01, 2.343750e+00, 1.101560e+00, -2.093750e+00]]> : tensor<5x7xbf16>
    return %0 : tensor<5x7xbf16>
  }
  func.func private @expected() -> tensor<5x7xbf16> {
    %0 = stablehlo.constant dense<[[-3.468750e+00, -1.960940e+00, 1.414060e+00, 1.351560e+00, 2.234380e+00, -1.726560e+00, -2.703130e+00], [-1.296880e+00, 1.281250e+00, 3.078130e+00, 6.171880e-01, 6.601560e-01, 3.718750e+00, 1.562500e+00], [-5.531250e+00, 3.734380e+00, 2.460940e-01, 1.867190e+00, -3.496090e-01, 4.062500e+00, 2.984380e+00], [1.570310e+00, 1.960940e+00, 9.531250e-01, -2.234380e+00, 2.609380e+00, -1.476560e+00, 4.433590e-01], [3.906250e+00, -6.591800e-02, -7.265630e-01, 3.437500e-01, 2.343750e+00, 1.101560e+00, -2.093750e+00]]> : tensor<5x7xbf16>
    return %0 : tensor<5x7xbf16>
  }
}
