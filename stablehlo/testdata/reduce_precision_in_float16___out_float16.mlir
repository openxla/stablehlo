// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<f16>
    %1 = call @expected() : () -> tensor<f16>
    %2 = stablehlo.reduce_precision %0, format = e5m10 : tensor<f16>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<f16>, tensor<f16>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<f16> {
    %0 = stablehlo.constant dense<6.743160e-01> : tensor<f16>
    return %0 : tensor<f16>
  }
  func.func private @expected() -> tensor<f16> {
    %0 = stablehlo.constant dense<6.743160e-01> : tensor<f16>
    return %0 : tensor<f16>
  }
}
