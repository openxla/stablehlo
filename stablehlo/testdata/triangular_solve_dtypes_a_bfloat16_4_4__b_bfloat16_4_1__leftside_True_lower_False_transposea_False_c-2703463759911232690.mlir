// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<4x4xbf16>, tensor<4x1xbf16>)
    %1 = call @expected() : () -> tensor<4x1xbf16>
    %2 = "stablehlo.triangular_solve"(%0#0, %0#1) {left_side = true, lower = false, transpose_a = #stablehlo<transpose NO_TRANSPOSE>, unit_diagonal = false} : (tensor<4x4xbf16>, tensor<4x1xbf16>) -> tensor<4x1xbf16>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<4x1xbf16>, tensor<4x1xbf16>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x4xbf16>, tensor<4x1xbf16>) {
    %0 = stablehlo.constant dense<[[-3.343750e+00, -1.031250e+00, -4.593750e+00, 1.617190e+00], [2.404790e-02, -2.207030e-01, -6.796880e-01, -3.031250e+00], [3.390630e+00, 1.242190e+00, -7.718750e+00, 4.093750e+00], [4.355470e-01, 2.828130e+00, 6.171880e-01, 3.593750e-01]]> : tensor<4x4xbf16>
    %1 = stablehlo.constant dense<[[-4.667970e-01], [2.484380e+00], [-9.648430e-01], [-2.406250e+00]]> : tensor<4x1xbf16>
    return %0, %1 : tensor<4x4xbf16>, tensor<4x1xbf16>
  }
  func.func private @expected() -> tensor<4x1xbf16> {
    %0 = stablehlo.constant dense<[[-2.662500e+01], [9.100000e+01], [-3.421880e+00], [-6.687500e+00]]> : tensor<4x1xbf16>
    return %0 : tensor<4x1xbf16>
  }
}
