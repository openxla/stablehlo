// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<3xf16>, tensor<3xf16>)
    %1 = call @expected() : () -> tensor<f32>
    %2 = stablehlo.convert %0#0 : (tensor<3xf16>) -> tensor<3xf32>
    %3 = stablehlo.convert %0#1 : (tensor<3xf16>) -> tensor<3xf32>
    %4 = "stablehlo.dot_general"(%2, %3) {dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [0], rhs_contracting_dimensions = [0]>} : (tensor<3xf32>, tensor<3xf32>) -> tensor<f32>
    %5 = stablehlo.custom_call @check.eq(%4, %1) : (tensor<f32>, tensor<f32>) -> tensor<i1>
    return %5 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<3xf16>, tensor<3xf16>) {
    %0 = stablehlo.constant dense<[1.627930e+00, -4.093750e+00, 1.308590e+00]> : tensor<3xf16>
    %1 = stablehlo.constant dense<[-1.496580e-01, 1.449220e+00, -7.140630e+00]> : tensor<3xf16>
    return %0, %1 : tensor<3xf16>, tensor<3xf16>
  }
  func.func private @expected() -> tensor<f32> {
    %0 = stablehlo.constant dense<-15.5205498> : tensor<f32>
    return %0 : tensor<f32>
  }
}

