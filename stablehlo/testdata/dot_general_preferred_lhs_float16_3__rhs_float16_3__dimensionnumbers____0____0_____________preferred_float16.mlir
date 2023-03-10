// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<3xf16>, tensor<3xf16>)
    %1 = call @expected() : () -> tensor<f16>
    %2 = stablehlo.convert %0#0 : (tensor<3xf16>) -> tensor<3xf32>
    %3 = stablehlo.convert %0#1 : (tensor<3xf16>) -> tensor<3xf32>
    %4 = "stablehlo.dot_general"(%2, %3) {dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [0], rhs_contracting_dimensions = [0]>} : (tensor<3xf32>, tensor<3xf32>) -> tensor<f16>
    %5 = stablehlo.custom_call @check.eq(%4, %1) : (tensor<f16>, tensor<f16>) -> tensor<i1>
    return %5 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<3xf16>, tensor<3xf16>) {
    %0 = stablehlo.constant dense<[-3.363280e+00, -3.357420e+00, 2.496090e+00]> : tensor<3xf16>
    %1 = stablehlo.constant dense<[-1.452150e+00, -3.039060e+00, 1.145630e-01]> : tensor<3xf16>
    return %0, %1 : tensor<3xf16>, tensor<3xf16>
  }
  func.func private @expected() -> tensor<f16> {
    %0 = stablehlo.constant dense<1.537500e+01> : tensor<f16>
    return %0 : tensor<f16>
  }
}

