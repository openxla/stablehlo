// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<3x3xbf16>, tensor<3x3xbf16>)
    %1 = call @expected() : () -> tensor<3x3xbf16>
    %2 = stablehlo.maximum %0#0, %0#1 : tensor<3x3xbf16>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<3x3xbf16>, tensor<3x3xbf16>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<3x3xbf16>, tensor<3x3xbf16>) {
    %0 = stablehlo.constant dense<[[0x7FC0, 0x7FC0, 0x7FC0], [0x7F80, 0x7F80, 0x7F80], [0xFF80, 0xFF80, 0xFF80]]> : tensor<3x3xbf16>
    %1 = stablehlo.constant dense<[[0x7FC0, 0x7F80, 0xFF80], [0x7FC0, 0x7F80, 0xFF80], [0x7FC0, 0x7F80, 0xFF80]]> : tensor<3x3xbf16>
    return %0, %1 : tensor<3x3xbf16>, tensor<3x3xbf16>
  }
  func.func private @expected() -> tensor<3x3xbf16> {
    %0 = stablehlo.constant dense<[[0x7FC0, 0x7FC0, 0x7FC0], [0x7FC0, 0x7F80, 0x7F80], [0x7FC0, 0x7F80, 0xFF80]]> : tensor<3x3xbf16>
    return %0 : tensor<3x3xbf16>
  }
}
