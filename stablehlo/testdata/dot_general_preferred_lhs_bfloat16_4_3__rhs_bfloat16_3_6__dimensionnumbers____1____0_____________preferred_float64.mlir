// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<4x3xbf16>, tensor<3x6xbf16>)
    %1 = call @expected() : () -> tensor<4x6xf32>
    %2 = "stablehlo.dot_general"(%0#0, %0#1) {dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<4x3xbf16>, tensor<3x6xbf16>) -> tensor<4x6xf32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<4x6xf32>, tensor<4x6xf32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x3xbf16>, tensor<3x6xbf16>) {
    %0 = stablehlo.constant dense<[[7.031250e-01, 1.187500e+00, 1.398440e+00], [-4.156250e+00, 1.484380e+00, 1.835940e+00], [1.632810e+00, 2.062500e+00, -2.750000e+00], [3.546880e+00, 3.156250e+00, 5.937500e-01]]> : tensor<4x3xbf16>
    %1 = stablehlo.constant dense<[[2.460940e-01, 2.156250e+00, -2.328130e+00, -5.981450e-02, 2.171880e+00, 5.234380e-01], [2.060550e-01, -2.871090e-01, 3.937500e+00, -3.109380e+00, -1.187500e+00, 2.046880e+00], [5.531250e+00, 1.523440e+00, -3.406250e+00, 4.343750e+00, 7.343750e-01, -4.156250e+00]]> : tensor<3x6xbf16>
    return %0, %1 : tensor<4x3xbf16>, tensor<3x6xbf16>
  }
  func.func private @expected() -> tensor<4x6xf32> {
    %0 = stablehlo.constant dense<[[8.15283203, 3.30560303, -1.72460938, 2.34002304, 1.1439209, -3.0135498], [9.43806457, -6.59115601, 9.26733398, 3.60797882, -9.44128417, -6.76782227], [-14.3841248, -1.26086426, 13.6868896, -18.4560642, -0.922485351, 16.5060425], [4.80740356, 7.64630126, 2.14770508, -7.44701767, 4.39135742, 5.84924316]]> : tensor<4x6xf32>
    return %0 : tensor<4x6xf32>
  }
}

