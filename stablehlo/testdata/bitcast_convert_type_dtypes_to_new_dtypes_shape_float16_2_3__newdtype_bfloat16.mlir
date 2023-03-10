// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<2x3xf16>
    %1 = call @expected() : () -> tensor<2x3xbf16>
    %2 = stablehlo.bitcast_convert %0 : (tensor<2x3xf16>) -> tensor<2x3xbf16>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<2x3xbf16>, tensor<2x3xbf16>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<2x3xf16> {
    %0 = stablehlo.constant dense<[[-6.997070e-01, -1.616210e-01, -4.089840e+00], [-3.713380e-01, 4.437500e+00, 3.085940e+00]]> : tensor<2x3xf16>
    return %0 : tensor<2x3xf16>
  }
  func.func private @expected() -> tensor<2x3xbf16> {
    %0 = stablehlo.constant dense<[[-2.918240e-04, -2.502930e-09, -6.040000e+02], [-1.795590e-06, 9.600000e+02, 4.300000e+01]]> : tensor<2x3xbf16>
    return %0 : tensor<2x3xbf16>
  }
}
