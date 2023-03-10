// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<3x4xf16>, tensor<4x2xf16>)
    %1 = call @expected() : () -> tensor<3x2xf16>
    %2 = stablehlo.convert %0#0 : (tensor<3x4xf16>) -> tensor<3x4xf32>
    %3 = stablehlo.convert %0#1 : (tensor<4x2xf16>) -> tensor<4x2xf32>
    %4 = "stablehlo.dot_general"(%2, %3) {dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>, precision_config = [#stablehlo<precision HIGH>, #stablehlo<precision HIGH>]} : (tensor<3x4xf32>, tensor<4x2xf32>) -> tensor<3x2xf16>
    %5 = stablehlo.custom_call @check.eq(%4, %1) : (tensor<3x2xf16>, tensor<3x2xf16>) -> tensor<i1>
    return %5 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<3x4xf16>, tensor<4x2xf16>) {
    %0 = stablehlo.constant dense<[[5.693360e-01, -4.296880e-01, -4.287110e-01, 1.562500e+00], [4.429690e+00, 1.643550e+00, 4.479980e-01, -2.265630e+00], [-1.674800e+00, 2.062500e+00, 2.066410e+00, 3.218750e+00]]> : tensor<3x4xf16>
    %1 = stablehlo.constant dense<[[-4.782710e-01, 1.126950e+00], [2.085880e-02, 4.500000e+00], [2.484130e-01, -3.275390e+00], [-1.822270e+00, -3.560550e+00]]> : tensor<4x2xf16>
    return %0, %1 : tensor<3x4xf16>, tensor<4x2xf16>
  }
  func.func private @expected() -> tensor<3x2xf16> {
    %0 = stablehlo.constant dense<[[-3.234380e+00, -5.449210e+00], [2.156250e+00, 1.898440e+01], [-4.507810e+00, -1.083590e+01]]> : tensor<3x2xf16>
    return %0 : tensor<3x2xf16>
  }
}
