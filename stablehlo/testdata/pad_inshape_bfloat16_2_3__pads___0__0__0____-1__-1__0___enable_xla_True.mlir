// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<2x3xbf16>, tensor<bf16>)
    %1 = call @expected() : () -> tensor<2x1xbf16>
    %2 = stablehlo.pad %0#0, %0#1, low = [0, -1], high = [0, -1], interior = [0, 0] : (tensor<2x3xbf16>, tensor<bf16>) -> tensor<2x1xbf16>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<2x1xbf16>, tensor<2x1xbf16>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<2x3xbf16>, tensor<bf16>) {
    %0 = stablehlo.constant dense<[[2.422330e-04, -1.152040e-03, 1.434330e-03], [5.531310e-04, -1.571660e-03, 1.182560e-03]]> : tensor<2x3xbf16>
    %1 = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    return %0, %1 : tensor<2x3xbf16>, tensor<bf16>
  }
  func.func private @expected() -> tensor<2x1xbf16> {
    %0 = stablehlo.constant dense<[[-1.152040e-03], [-1.571660e-03]]> : tensor<2x1xbf16>
    return %0 : tensor<2x1xbf16>
  }
}
