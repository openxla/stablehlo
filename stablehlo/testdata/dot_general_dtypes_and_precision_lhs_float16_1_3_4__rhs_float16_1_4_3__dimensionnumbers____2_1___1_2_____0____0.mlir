// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<1x3x4xf16>, tensor<1x4x3xf16>)
    %1 = call @expected() : () -> tensor<1xf16>
    %2 = stablehlo.convert %0#0 : (tensor<1x3x4xf16>) -> tensor<1x3x4xf32>
    %3 = stablehlo.convert %0#1 : (tensor<1x4x3xf16>) -> tensor<1x4x3xf32>
    %4 = "stablehlo.dot_general"(%2, %3) {dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2, 1], rhs_contracting_dimensions = [1, 2]>} : (tensor<1x3x4xf32>, tensor<1x4x3xf32>) -> tensor<1xf16>
    %5 = stablehlo.custom_call @check.eq(%4, %1) : (tensor<1xf16>, tensor<1xf16>) -> tensor<i1>
    return %5 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<1x3x4xf16>, tensor<1x4x3xf16>) {
    %0 = stablehlo.constant dense<[[[-5.437500e+00, 9.375000e+00, -4.449220e+00, 6.333010e-01], [3.001950e+00, 2.806640e+00, 3.894530e+00, 2.255860e-01], [2.580570e-01, -1.491210e+00, 7.978520e-01, 1.056640e+00]]]> : tensor<1x3x4xf16>
    %1 = stablehlo.constant dense<[[[-3.859380e+00, 4.355470e+00, -9.785150e-01], [6.265630e+00, 3.910160e+00, 1.794920e+00], [5.834960e-02, -3.794920e+00, 1.802730e+00], [2.565920e-01, 3.919920e+00, -3.244140e+00]]]> : tensor<1x4x3xf16>
    return %0, %1 : tensor<1x3x4xf16>, tensor<1x4x3xf16>
  }
  func.func private @expected() -> tensor<1xf16> {
    %0 = stablehlo.constant dense<8.487500e+01> : tensor<1xf16>
    return %0 : tensor<1xf16>
  }
}

