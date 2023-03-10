// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<2xf16>, tensor<2xf16>)
    %1 = call @expected() : () -> tensor<2xf16>
    %2 = stablehlo.add %0#0, %0#1 : tensor<2xf16>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<2xf16>, tensor<2xf16>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<2xf16>, tensor<2xf16>) {
    %0 = stablehlo.constant dense<[3.617190e+00, 1.857420e+00]> : tensor<2xf16>
    %1 = stablehlo.constant dense<[-2.578130e+00, 3.662110e-03]> : tensor<2xf16>
    return %0, %1 : tensor<2xf16>, tensor<2xf16>
  }
  func.func private @expected() -> tensor<2xf16> {
    %0 = stablehlo.constant dense<[1.039060e+00, 1.861330e+00]> : tensor<2xf16>
    return %0 : tensor<2xf16>
  }
}
