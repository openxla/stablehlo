// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<4x4xf16>, tensor<4x1xf16>)
    %1 = call @expected() : () -> tensor<4x1xf16>
    %2 = "stablehlo.triangular_solve"(%0#0, %0#1) {left_side = true, lower = false, transpose_a = #stablehlo<transpose NO_TRANSPOSE>, unit_diagonal = false} : (tensor<4x4xf16>, tensor<4x1xf16>) -> tensor<4x1xf16>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<4x1xf16>, tensor<4x1xf16>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x4xf16>, tensor<4x1xf16>) {
    %0 = stablehlo.constant dense<[[-4.221190e-01, 1.652830e-01, 1.440430e+00, 2.916020e+00], [3.531250e+00, 2.404300e+00, -1.565430e+00, 2.076170e+00], [-3.367190e+00, 7.397460e-01, -3.664550e-01, 1.849610e+00], [-2.814450e+00, -8.984370e-01, -4.750000e+00, 2.724610e+00]]> : tensor<4x4xf16>
    %1 = stablehlo.constant dense<[[-3.488280e+00], [-2.119140e+00], [6.953130e+00], [8.808590e-01]]> : tensor<4x1xf16>
    return %0, %1 : tensor<4x4xf16>, tensor<4x1xf16>
  }
  func.func private @expected() -> tensor<4x1xf16> {
    %0 = stablehlo.constant dense<[[-5.362500e+01], [-1.245310e+01], [-1.734380e+01], [3.232420e-01]]> : tensor<4x1xf16>
    return %0 : tensor<4x1xf16>
  }
}
