// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<4x3xf16>, tensor<3xf16>)
    %1 = call @expected() : () -> tensor<4xf32>
    %2 = stablehlo.convert %0#0 : (tensor<4x3xf16>) -> tensor<4x3xf32>
    %3 = stablehlo.convert %0#1 : (tensor<3xf16>) -> tensor<3xf32>
    %4 = "stablehlo.dot_general"(%2, %3) {dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<4x3xf32>, tensor<3xf32>) -> tensor<4xf32>
    %5 = stablehlo.custom_call @check.eq(%4, %1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<i1>
    return %5 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x3xf16>, tensor<3xf16>) {
    %0 = stablehlo.constant dense<[[-4.316410e+00, -5.395510e-01, -8.613280e-01], [2.029300e+00, -4.658200e-01, -4.042970e+00], [-2.083980e+00, -2.136720e+00, 2.888670e+00], [-1.613280e+00, 3.908200e+00, -1.158590e+01]]> : tensor<4x3xf16>
    %1 = stablehlo.constant dense<[-1.079100e+00, 2.287110e+00, 4.386720e+00]> : tensor<3xf16>
    return %0, %1 : tensor<4x3xf16>, tensor<3xf16>
  }
  func.func private @expected() -> tensor<4xf32> {
    %0 = stablehlo.constant dense<[-0.354575157, -20.9905663, 10.0337124, -40.1448669]> : tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}

