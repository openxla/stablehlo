// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<3xf16>, tensor<3x6xf16>)
    %1 = call @expected() : () -> tensor<6xf16>
    %2 = stablehlo.convert %0#0 : (tensor<3xf16>) -> tensor<3xf32>
    %3 = stablehlo.convert %0#1 : (tensor<3x6xf16>) -> tensor<3x6xf32>
    %4 = "stablehlo.dot_general"(%2, %3) {dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [0], rhs_contracting_dimensions = [0]>} : (tensor<3xf32>, tensor<3x6xf32>) -> tensor<6xf16>
    %5 = stablehlo.custom_call @check.eq(%4, %1) : (tensor<6xf16>, tensor<6xf16>) -> tensor<i1>
    return %5 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<3xf16>, tensor<3x6xf16>) {
    %0 = stablehlo.constant dense<[-6.084440e-03, 4.011720e+00, -7.617180e-01]> : tensor<3xf16>
    %1 = stablehlo.constant dense<[[1.291020e+00, -1.114260e+00, 4.605470e+00, -6.265630e+00, -5.214840e-01, -9.497070e-01], [-1.554690e+00, -3.626950e+00, -3.087890e+00, 6.586910e-01, -7.939450e-01, 1.101560e+00], [5.366210e-01, -3.802730e+00, -1.345700e+00, 2.556150e-01, -3.449220e+00, 8.461910e-01]]> : tensor<3x6xf16>
    return %0, %1 : tensor<3xf16>, tensor<3x6xf16>
  }
  func.func private @expected() -> tensor<6xf16> {
    %0 = stablehlo.constant dense<[-6.652340e+00, -1.164840e+01, -1.139060e+01, 2.486330e+00, -5.546880e-01, 3.781250e+00]> : tensor<6xf16>
    return %0 : tensor<6xf16>
  }
}

