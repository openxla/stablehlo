// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<2x3xbf16>, tensor<bf16>)
    %1 = call @expected() : () -> tensor<6x4xbf16>
    %2 = stablehlo.pad %0#0, %0#1, low = [1, 0], high = [2, 1], interior = [1, 0] : (tensor<2x3xbf16>, tensor<bf16>) -> tensor<6x4xbf16>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<6x4xbf16>, tensor<6x4xbf16>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<2x3xbf16>, tensor<bf16>) {
    %0 = stablehlo.constant dense<[[-1.907350e-04, 7.591240e-04, 4.997250e-04], [2.288820e-04, -6.790160e-04, -1.850130e-04]]> : tensor<2x3xbf16>
    %1 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    return %0, %1 : tensor<2x3xbf16>, tensor<bf16>
  }
  func.func private @expected() -> tensor<6x4xbf16> {
    %0 = stablehlo.constant dense<[[0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [-1.907350e-04, 7.591240e-04, 4.997250e-04, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [2.288820e-04, -6.790160e-04, -1.850130e-04, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]]> : tensor<6x4xbf16>
    return %0 : tensor<6x4xbf16>
  }
}
