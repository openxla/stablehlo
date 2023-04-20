// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<3x4xf16>, tensor<4x2xf16>)
    %1 = call @expected() : () -> tensor<3x2xf16>
    %2 = stablehlo.convert %0#0 : (tensor<3x4xf16>) -> tensor<3x4xf32>
    %3 = stablehlo.convert %0#1 : (tensor<4x2xf16>) -> tensor<4x2xf32>
    %4 = "stablehlo.dot_general"(%2, %3) {dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<3x4xf32>, tensor<4x2xf32>) -> tensor<3x2xf16>
    %5 = stablehlo.custom_call @check.eq(%4, %1) : (tensor<3x2xf16>, tensor<3x2xf16>) -> tensor<i1>
    return %5 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<3x4xf16>, tensor<4x2xf16>) {
    %0 = stablehlo.constant dense<[[2.312500e+00, -1.579100e+00, -2.056640e+00, 3.484380e+00], [3.791020e+00, 8.012690e-01, 4.437500e+00, -1.961910e+00], [1.671880e+00, -7.934570e-01, -1.700200e+00, 2.097660e+00]]> : tensor<3x4xf16>
    %1 = stablehlo.constant dense<[[1.937500e+00, -1.952150e+00], [-3.728520e+00, 2.692870e-01], [-2.535160e+00, -2.833980e+00], [-5.593750e+00, 2.283200e+00]]> : tensor<4x2xf16>
    return %0, %1 : tensor<3x4xf16>, tensor<4x2xf16>
  }
  func.func private @expected() -> tensor<3x2xf16> {
    %0 = stablehlo.constant dense<[[-3.908200e+00, 8.843750e+00], [4.082030e+00, -2.423440e+01], [-1.225590e+00, 6.128900e+00]]> : tensor<3x2xf16>
    return %0 : tensor<3x2xf16>
  }
}

