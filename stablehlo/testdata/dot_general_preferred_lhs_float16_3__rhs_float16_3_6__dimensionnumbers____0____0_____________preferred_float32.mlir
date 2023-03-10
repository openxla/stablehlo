// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<3xf16>, tensor<3x6xf16>)
    %1 = call @expected() : () -> tensor<6xf32>
    %2 = stablehlo.convert %0#0 : (tensor<3xf16>) -> tensor<3xf32>
    %3 = stablehlo.convert %0#1 : (tensor<3x6xf16>) -> tensor<3x6xf32>
    %4 = "stablehlo.dot_general"(%2, %3) {dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [0], rhs_contracting_dimensions = [0]>} : (tensor<3xf32>, tensor<3x6xf32>) -> tensor<6xf32>
    %5 = stablehlo.custom_call @check.eq(%4, %1) : (tensor<6xf32>, tensor<6xf32>) -> tensor<i1>
    return %5 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<3xf16>, tensor<3x6xf16>) {
    %0 = stablehlo.constant dense<[-1.288090e+00, -2.345700e+00, -1.739260e+00]> : tensor<3xf16>
    %1 = stablehlo.constant dense<[[-7.160150e+00, -1.794920e+00, 5.644530e-01, -2.544920e+00, -1.419680e-01, 5.230470e+00], [4.210940e+00, -4.406250e+00, 3.347660e+00, -2.394530e+00, -1.793950e+00, -4.300780e+00], [6.726560e+00, -6.630860e-01, 4.253910e+00, 2.496090e+00, -2.978520e+00, 1.707030e+00]]> : tensor<3x6xf16>
    return %0, %1 : tensor<3xf16>, tensor<3x6xf16>
  }
  func.func private @expected() -> tensor<6xf32> {
    %0 = stablehlo.constant dense<[-12.3539391, 13.8010454, -15.9783115, 4.55358696, 9.57133674, 0.382095337]> : tensor<6xf32>
    return %0 : tensor<6xf32>
  }
}

