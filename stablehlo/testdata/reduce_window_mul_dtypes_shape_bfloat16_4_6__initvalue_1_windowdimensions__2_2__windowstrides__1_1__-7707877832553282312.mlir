// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<4x6xbf16>
    %1 = call @expected() : () -> tensor<3x5xbf16>
    %2 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %3 = "stablehlo.reduce_window"(%0, %2) ({
    ^bb0(%arg0: tensor<bf16>, %arg1: tensor<bf16>):
      %5 = stablehlo.multiply %arg0, %arg1 : tensor<bf16>
      stablehlo.return %5 : tensor<bf16>
    }) {window_dimensions = dense<2> : tensor<2xi64>} : (tensor<4x6xbf16>, tensor<bf16>) -> tensor<3x5xbf16>
    %4 = stablehlo.custom_call @check.eq(%3, %1) : (tensor<3x5xbf16>, tensor<3x5xbf16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> tensor<4x6xbf16> {
    %0 = stablehlo.constant dense<[[-1.726560e+00, -1.351560e+00, 1.882810e+00, 6.312500e+00, -4.257810e-01, 5.187500e+00], [-1.476560e+00, 4.812500e+00, -2.171880e+00, 2.171880e+00, -1.269530e-01, -3.710940e-01], [-5.390630e-01, -1.757810e-01, 7.070310e-01, -1.109380e+00, -6.523440e-01, 8.837890e-02], [9.375000e-01, 4.468750e+00, -4.843750e+00, -1.757810e+00, 1.906250e+00, -1.273440e+00]]> : tensor<4x6xbf16>
    return %0 : tensor<4x6xbf16>
  }
  func.func private @expected() -> tensor<3x5xbf16> {
    %0 = stablehlo.constant dense<[[-1.650000e+01, 2.662500e+01, -5.600000e+01, 7.421880e-01, -1.035160e-01], [-6.718750e-01, 1.296880e+00, 3.703130e+00, -1.992190e-01, -2.716060e-03], [3.964840e-01, 2.687500e+00, -6.687500e+00, -2.421880e+00, 1.396480e-01]]> : tensor<3x5xbf16>
    return %0 : tensor<3x5xbf16>
  }
}

