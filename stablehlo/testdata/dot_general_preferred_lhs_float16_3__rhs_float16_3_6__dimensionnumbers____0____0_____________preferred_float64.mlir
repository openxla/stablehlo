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
    %0 = stablehlo.constant dense<[2.980960e-01, 1.846920e-01, -1.654300e+00]> : tensor<3xf16>
    %1 = stablehlo.constant dense<[[-3.582030e+00, -2.082030e+00, -1.664060e+00, -1.207280e-01, 3.867190e+00, 1.593750e+00], [-1.520510e+00, -3.005860e+00, -8.852530e-01, 1.785160e+00, -2.856450e-01, -7.993160e-01], [4.410160e+00, 2.425780e+00, -9.062500e-01, -4.281250e+00, 1.187130e-01, -2.496090e+00]]> : tensor<3x6xf16>
    return %0, %1 : tensor<3xf16>, tensor<3x6xf16>
  }
  func.func private @expected() -> tensor<6xf32> {
    %0 = stablehlo.constant dense<[-8.64432239, -5.18876648, 8.396570e-01, 7.37617492, 0.903648495, 4.45674229]> : tensor<6xf32>
    return %0 : tensor<6xf32>
  }
}

