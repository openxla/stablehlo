// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<4x4xbf16>, tensor<4x1xbf16>)
    %1 = call @expected() : () -> tensor<4x1xbf16>
    %2 = "stablehlo.triangular_solve"(%0#0, %0#1) {left_side = true, lower = false, transpose_a = #stablehlo<transpose NO_TRANSPOSE>, unit_diagonal = true} : (tensor<4x4xbf16>, tensor<4x1xbf16>) -> tensor<4x1xbf16>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<4x1xbf16>, tensor<4x1xbf16>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x4xbf16>, tensor<4x1xbf16>) {
    %0 = stablehlo.constant dense<[[2.921880e+00, -2.296880e+00, 2.687500e+00, -2.695310e-01], [4.414060e-01, -1.726560e+00, 2.562500e+00, 2.609380e+00], [-1.593750e+00, -2.703130e+00, 1.289060e-01, 1.804690e+00], [-1.250000e+00, -4.028320e-02, 2.390630e+00, 1.445310e-01]]> : tensor<4x4xbf16>
    %1 = stablehlo.constant dense<[[-1.187500e+00], [-3.015630e+00], [-1.632810e+00], [-3.593750e+00]]> : tensor<4x1xbf16>
    return %0, %1 : tensor<4x4xbf16>, tensor<4x1xbf16>
  }
  func.func private @expected() -> tensor<4x1xbf16> {
    %0 = stablehlo.constant dense<[[-2.912500e+01], [-6.062500e+00], [4.843750e+00], [-3.593750e+00]]> : tensor<4x1xbf16>
    return %0 : tensor<4x1xbf16>
  }
}
