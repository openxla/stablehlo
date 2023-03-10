// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<4x3xf16>, tensor<3x6xf16>)
    %1 = call @expected() : () -> tensor<4x6xf16>
    %2 = stablehlo.convert %0#0 : (tensor<4x3xf16>) -> tensor<4x3xf32>
    %3 = stablehlo.convert %0#1 : (tensor<3x6xf16>) -> tensor<3x6xf32>
    %4 = "stablehlo.dot_general"(%2, %3) {dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<4x3xf32>, tensor<3x6xf32>) -> tensor<4x6xf16>
    %5 = stablehlo.custom_call @check.eq(%4, %1) : (tensor<4x6xf16>, tensor<4x6xf16>) -> tensor<i1>
    return %5 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x3xf16>, tensor<3x6xf16>) {
    %0 = stablehlo.constant dense<[[5.867190e+00, 2.750000e+00, 1.076170e+00], [1.553710e+00, -4.910160e+00, -1.748050e+00], [-6.351560e+00, 3.098140e-01, 3.806640e+00], [2.736330e+00, -2.763670e-01, 4.324220e+00]]> : tensor<4x3xf16>
    %1 = stablehlo.constant dense<[[4.093750e+00, -6.263730e-03, -1.127930e+00, -2.482420e+00, -2.548830e+00, 3.529300e+00], [2.431640e+00, 2.136230e-01, 1.166990e+00, -2.533200e+00, -6.347650e-01, -1.823240e+00], [2.119140e+00, -3.847660e+00, -2.789060e+00, 4.765630e+00, -1.204100e+00, -3.863280e+00]]> : tensor<3x6xf16>
    return %0, %1 : tensor<4x3xf16>, tensor<3x6xf16>
  }
  func.func private @expected() -> tensor<4x6xf16> {
    %0 = stablehlo.constant dense<[[3.300000e+01, -3.589840e+00, -6.410150e+00, -1.640630e+01, -1.800000e+01, 1.153910e+01], [-9.281250e+00, 5.667960e+00, -2.607420e+00, 2.509770e-01, 1.261720e+00, 2.118750e+01], [-1.718750e+01, -1.453910e+01, -3.091800e+00, 3.312500e+01, 1.140630e+01, -3.768750e+01], [1.968750e+01, -1.671880e+01, -1.546880e+01, 1.451560e+01, -1.200780e+01, -6.542960e+00]]> : tensor<4x6xf16>
    return %0 : tensor<4x6xf16>
  }
}

