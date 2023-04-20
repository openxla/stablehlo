// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<2x3xf16>
    %1 = call @expected() : () -> tensor<2x3xf16>
    %2 = stablehlo.bitcast_convert %0 : (tensor<2x3xf16>) -> tensor<2x3xf16>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<2x3xf16>, tensor<2x3xf16>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<2x3xf16> {
    %0 = stablehlo.constant dense<[[-2.441410e+00, -1.168950e+00, -1.477540e+00], [-1.548830e+00, 2.732420e+00, -5.595700e-01]]> : tensor<2x3xf16>
    return %0 : tensor<2x3xf16>
  }
  func.func private @expected() -> tensor<2x3xf16> {
    %0 = stablehlo.constant dense<[[-2.441410e+00, -1.168950e+00, -1.477540e+00], [-1.548830e+00, 2.732420e+00, -5.595700e-01]]> : tensor<2x3xf16>
    return %0 : tensor<2x3xf16>
  }
}
