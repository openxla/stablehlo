// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<4x3xbf16>, tensor<3xbf16>)
    %1 = call @expected() : () -> tensor<4xf32>
    %2 = "stablehlo.dot_general"(%0#0, %0#1) {dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<4x3xbf16>, tensor<3xbf16>) -> tensor<4xf32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x3xbf16>, tensor<3xbf16>) {
    %0 = stablehlo.constant dense<[[-1.435550e-01, -3.390630e+00, -6.281250e+00], [-2.484380e+00, -3.140630e+00, 5.156250e-01], [-2.156250e+00, 2.480470e-01, -9.492180e-01], [-2.656250e+00, -1.240230e-01, 3.476560e-01]]> : tensor<4x3xbf16>
    %1 = stablehlo.constant dense<[2.234380e+00, 2.750000e+00, -4.687500e+00]> : tensor<3xbf16>
    return %0, %1 : tensor<4x3xbf16>, tensor<3xbf16>
  }
  func.func private @expected() -> tensor<4xf32> {
    %0 = stablehlo.constant dense<[19.7983856, -16.6047363, 0.313720703, -7.90576171]> : tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}

