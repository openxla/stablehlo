// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<2x3xf16>, tensor<2x3xf16>)
    %1 = call @expected() : () -> tensor<4x3xf16>
    %2 = stablehlo.concatenate %0#0, %0#1, dim = 0 : (tensor<2x3xf16>, tensor<2x3xf16>) -> tensor<4x3xf16>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<4x3xf16>, tensor<4x3xf16>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<2x3xf16>, tensor<2x3xf16>) {
    %0 = stablehlo.constant dense<[[1.591800e-01, -1.018550e+00, 3.117190e+00], [4.136720e+00, 3.169920e+00, -1.167970e+00]]> : tensor<2x3xf16>
    %1 = stablehlo.constant dense<[[2.332030e+00, -2.121090e+00, -5.816400e+00], [-1.439450e+00, 8.078130e+00, -1.614260e+00]]> : tensor<2x3xf16>
    return %0, %1 : tensor<2x3xf16>, tensor<2x3xf16>
  }
  func.func private @expected() -> tensor<4x3xf16> {
    %0 = stablehlo.constant dense<[[1.591800e-01, -1.018550e+00, 3.117190e+00], [4.136720e+00, 3.169920e+00, -1.167970e+00], [2.332030e+00, -2.121090e+00, -5.816400e+00], [-1.439450e+00, 8.078130e+00, -1.614260e+00]]> : tensor<4x3xf16>
    return %0 : tensor<4x3xf16>
  }
}
