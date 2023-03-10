// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<5x7xbf16>
    %1 = call @expected() : () -> tensor<5x7xbf16>
    %2 = stablehlo.reduce_precision %0, format = e5m10 : tensor<5x7xbf16>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<5x7xbf16>, tensor<5x7xbf16>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<5x7xbf16> {
    %0 = stablehlo.constant dense<[[6.396480e-02, -5.898440e-01, -1.765630e+00, -1.437500e+00, 3.613280e-01, -4.179690e-01, 5.117190e-01], [2.078130e+00, 2.636720e-01, 7.226560e-01, 2.578130e-01, -9.687500e+00, 1.312500e+00, 3.781250e+00], [1.421880e+00, -1.757810e+00, 1.750000e+00, -2.437500e+00, 2.226560e-01, -1.070310e+00, 5.937500e+00], [-3.687500e+00, -1.781250e+00, -3.500000e+00, -2.812500e+00, -1.078130e+00, 4.406250e+00, 2.375000e+00], [1.828130e+00, -2.062500e+00, -3.421880e+00, 2.093750e+00, 2.109380e+00, 2.046880e+00, 1.789060e+00]]> : tensor<5x7xbf16>
    return %0 : tensor<5x7xbf16>
  }
  func.func private @expected() -> tensor<5x7xbf16> {
    %0 = stablehlo.constant dense<[[6.396480e-02, -5.898440e-01, -1.765630e+00, -1.437500e+00, 3.613280e-01, -4.179690e-01, 5.117190e-01], [2.078130e+00, 2.636720e-01, 7.226560e-01, 2.578130e-01, -9.687500e+00, 1.312500e+00, 3.781250e+00], [1.421880e+00, -1.757810e+00, 1.750000e+00, -2.437500e+00, 2.226560e-01, -1.070310e+00, 5.937500e+00], [-3.687500e+00, -1.781250e+00, -3.500000e+00, -2.812500e+00, -1.078130e+00, 4.406250e+00, 2.375000e+00], [1.828130e+00, -2.062500e+00, -3.421880e+00, 2.093750e+00, 2.109380e+00, 2.046880e+00, 1.789060e+00]]> : tensor<5x7xbf16>
    return %0 : tensor<5x7xbf16>
  }
}
