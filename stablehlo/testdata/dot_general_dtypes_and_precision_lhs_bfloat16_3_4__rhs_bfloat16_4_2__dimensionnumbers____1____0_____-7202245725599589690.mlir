// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<3x4xbf16>, tensor<4x2xbf16>)
    %1 = call @expected() : () -> tensor<3x2xbf16>
    %2 = "stablehlo.dot_general"(%0#0, %0#1) {dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>} : (tensor<3x4xbf16>, tensor<4x2xbf16>) -> tensor<3x2xbf16>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<3x2xbf16>, tensor<3x2xbf16>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<3x4xbf16>, tensor<4x2xbf16>) {
    %0 = stablehlo.constant dense<[[-4.687500e+00, 3.843750e+00, 3.359380e+00, 6.812500e+00], [4.031250e+00, -2.093750e+00, -1.671880e+00, 2.636720e-01], [-1.625000e+00, 4.199220e-02, 2.656250e+00, -3.343750e+00]]> : tensor<3x4xbf16>
    %1 = stablehlo.constant dense<[[-1.117190e+00, 3.578130e+00], [-4.511720e-01, -2.031250e-01], [-2.255860e-01, -9.257810e-01], [6.328130e-01, -4.593750e+00]]> : tensor<4x2xbf16>
    return %0, %1 : tensor<3x4xbf16>, tensor<4x2xbf16>
  }
  func.func private @expected() -> tensor<3x2xbf16> {
    %0 = stablehlo.constant dense<[[7.062500e+00, -5.200000e+01], [-3.015630e+00, 1.518750e+01], [-9.179680e-01, 7.093750e+00]]> : tensor<3x2xbf16>
    return %0 : tensor<3x2xbf16>
  }
}

