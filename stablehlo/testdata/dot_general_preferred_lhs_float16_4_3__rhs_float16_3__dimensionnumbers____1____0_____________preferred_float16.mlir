// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<4x3xf16>, tensor<3xf16>)
    %1 = call @expected() : () -> tensor<4xf16>
    %2 = stablehlo.convert %0#0 : (tensor<4x3xf16>) -> tensor<4x3xf32>
    %3 = stablehlo.convert %0#1 : (tensor<3xf16>) -> tensor<3xf32>
    %4 = "stablehlo.dot_general"(%2, %3) {dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<4x3xf32>, tensor<3xf32>) -> tensor<4xf16>
    %5 = stablehlo.custom_call @check.eq(%4, %1) : (tensor<4xf16>, tensor<4xf16>) -> tensor<i1>
    return %5 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x3xf16>, tensor<3xf16>) {
    %0 = stablehlo.constant dense<[[-6.558590e+00, 1.823240e+00, -1.149410e+00], [9.289060e+00, 4.093750e+00, 1.858400e+00], [4.648440e+00, 2.576170e+00, -3.029300e+00], [3.191410e+00, -1.561520e+00, 8.959960e-01]]> : tensor<4x3xf16>
    %1 = stablehlo.constant dense<[4.757810e+00, -5.678710e-01, 5.214840e+00]> : tensor<3xf16>
    return %0, %1 : tensor<4x3xf16>, tensor<3xf16>
  }
  func.func private @expected() -> tensor<4xf16> {
    %0 = stablehlo.constant dense<[-3.821880e+01, 5.156250e+01, 4.855470e+00, 2.075000e+01]> : tensor<4xf16>
    return %0 : tensor<4xf16>
  }
}

