// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<4x4xf16>, tensor<4x1xf16>)
    %1 = call @expected() : () -> tensor<4x1xf16>
    %2 = "stablehlo.triangular_solve"(%0#0, %0#1) {left_side = true, lower = false, transpose_a = #stablehlo<transpose NO_TRANSPOSE>, unit_diagonal = true} : (tensor<4x4xf16>, tensor<4x1xf16>) -> tensor<4x1xf16>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<4x1xf16>, tensor<4x1xf16>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x4xf16>, tensor<4x1xf16>) {
    %0 = stablehlo.constant dense<[[-1.334960e+00, -2.302730e+00, 4.351560e+00, -2.498050e+00], [3.691410e-01, -1.691410e+00, -3.259770e+00, -2.955080e+00], [1.032230e+00, 3.697270e+00, -6.972650e+00, -1.898440e+00], [-1.192380e+00, 3.990230e+00, 3.474610e+00, 1.678470e-01]]> : tensor<4x4xf16>
    %1 = stablehlo.constant dense<[[-2.558590e-01], [-1.599610e+00], [-3.517580e+00], [-3.352050e-01]]> : tensor<4x1xf16>
    return %0, %1 : tensor<4x4xf16>, tensor<4x1xf16>
  }
  func.func private @expected() -> tensor<4x1xf16> {
    %0 = stablehlo.constant dense<[[-2.015630e+01], [-1.612500e+01], [-4.152340e+00], [-3.352050e-01]]> : tensor<4x1xf16>
    return %0 : tensor<4x1xf16>
  }
}
