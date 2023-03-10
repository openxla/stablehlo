// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<2x3xbf16>, tensor<bf16>)
    %1 = call @expected() : () -> tensor<2x3xbf16>
    %2 = stablehlo.pad %0#0, %0#1, low = [0, 0], high = [0, 0], interior = [0, 0] : (tensor<2x3xbf16>, tensor<bf16>) -> tensor<2x3xbf16>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<2x3xbf16>, tensor<2x3xbf16>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<2x3xbf16>, tensor<bf16>) {
    %0 = stablehlo.constant dense<[[-1.060490e-03, 1.697540e-04, -6.246570e-05], [-1.846310e-03, 5.626680e-05, 2.069470e-04]]> : tensor<2x3xbf16>
    %1 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    return %0, %1 : tensor<2x3xbf16>, tensor<bf16>
  }
  func.func private @expected() -> tensor<2x3xbf16> {
    %0 = stablehlo.constant dense<[[-1.060490e-03, 1.697540e-04, -6.246570e-05], [-1.846310e-03, 5.626680e-05, 2.069470e-04]]> : tensor<2x3xbf16>
    return %0 : tensor<2x3xbf16>
  }
}
