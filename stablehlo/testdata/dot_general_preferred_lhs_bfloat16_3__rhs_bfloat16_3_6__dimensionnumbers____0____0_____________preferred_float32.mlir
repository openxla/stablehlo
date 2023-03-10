// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<3xbf16>, tensor<3x6xbf16>)
    %1 = call @expected() : () -> tensor<6xf32>
    %2 = "stablehlo.dot_general"(%0#0, %0#1) {dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [0], rhs_contracting_dimensions = [0]>} : (tensor<3xbf16>, tensor<3x6xbf16>) -> tensor<6xf32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<6xf32>, tensor<6xf32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<3xbf16>, tensor<3x6xbf16>) {
    %0 = stablehlo.constant dense<[-1.593750e+00, 6.968750e+00, -1.500000e+00]> : tensor<3xbf16>
    %1 = stablehlo.constant dense<[[-3.750000e-01, 4.570310e-01, 2.275390e-01, -4.968750e+00, 1.960940e+00, 1.671880e+00], [1.148440e+00, -9.648430e-01, 2.558590e-01, 1.203130e+00, -2.937500e+00, 1.460940e+00], [3.156250e+00, -1.904300e-01, -1.335940e+00, -4.406250e+00, 8.281250e-01, -4.156250e+00]]> : tensor<3x6xbf16>
    return %0, %1 : tensor<3xbf16>, tensor<3x6xbf16>
  }
  func.func private @expected() -> tensor<6xf32> {
    %0 = stablehlo.constant dense<[3.86645508, -7.1665039, 3.42428589, 22.9125977, -24.8381348, 13.7507324]> : tensor<6xf32>
    return %0 : tensor<6xf32>
  }
}

