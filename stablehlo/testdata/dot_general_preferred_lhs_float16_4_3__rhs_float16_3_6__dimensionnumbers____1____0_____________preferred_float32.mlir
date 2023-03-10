// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<4x3xf16>, tensor<3x6xf16>)
    %1 = call @expected() : () -> tensor<4x6xf32>
    %2 = stablehlo.convert %0#0 : (tensor<4x3xf16>) -> tensor<4x3xf32>
    %3 = stablehlo.convert %0#1 : (tensor<3x6xf16>) -> tensor<3x6xf32>
    %4 = "stablehlo.dot_general"(%2, %3) {dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<4x3xf32>, tensor<3x6xf32>) -> tensor<4x6xf32>
    %5 = stablehlo.custom_call @check.eq(%4, %1) : (tensor<4x6xf32>, tensor<4x6xf32>) -> tensor<i1>
    return %5 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x3xf16>, tensor<3x6xf16>) {
    %0 = stablehlo.constant dense<[[5.004880e-01, 3.671880e+00, -5.242190e+00], [-2.408200e+00, -4.906250e+00, -4.171880e+00], [-4.255370e-01, 1.076170e+00, 6.699210e-01], [4.105470e+00, -4.710940e+00, -2.177730e+00]]> : tensor<4x3xf16>
    %1 = stablehlo.constant dense<[[-9.536130e-01, 6.343750e+00, -4.910160e+00, -9.375000e-01, 1.521480e+00, -1.712890e+00], [2.775880e-01, 4.748540e-01, -4.476560e+00, -1.804690e+00, 3.947270e+00, 1.227540e+00], [8.535150e-01, 2.287110e+00, -8.979490e-01, 2.341800e+00, -1.369630e-01, -5.458980e-01]]> : tensor<3x6xf16>
    return %0, %1 : tensor<4x3xf16>, tensor<3x6xf16>
  }
  func.func private @expected() -> tensor<4x6xf32> {
    %0 = stablehlo.constant dense<[[-3.93229318, -7.07088089, -14.1876354, -19.371933, 15.9733362, 6.51179027], [-2.6261816, -27.1483231, 37.5339203, 1.34225464, -22.4589233, 0.379795074], [1.27631891, -0.656292438, -3.32965279, 0.0256080627, 3.50873375, 1.68423223], [-7.08145904, 18.8263454, 2.88580799, -0.446918488, -12.0506458, -11.6262569]]> : tensor<4x6xf32>
    return %0 : tensor<4x6xf32>
  }
}

