// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<2x3xui16>
    %1 = call @expected() : () -> tensor<3xui16>
    %2 = stablehlo.constant dense<65535> : tensor<ui16>
    %3 = stablehlo.reduce(%0 init: %2) across dimensions = [0] : (tensor<2x3xui16>, tensor<ui16>) -> tensor<3xui16>
     reducer(%arg0: tensor<ui16>, %arg1: tensor<ui16>)  {
      %5 = stablehlo.minimum %arg0, %arg1 : tensor<ui16>
      stablehlo.return %5 : tensor<ui16>
    }
    %4 = stablehlo.custom_call @check.eq(%3, %1) : (tensor<3xui16>, tensor<3xui16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> tensor<2x3xui16> {
    %0 = stablehlo.constant dense<[[5, 0, 1], [0, 0, 1]]> : tensor<2x3xui16>
    return %0 : tensor<2x3xui16>
  }
  func.func private @expected() -> tensor<3xui16> {
    %0 = stablehlo.constant dense<[0, 0, 1]> : tensor<3xui16>
    return %0 : tensor<3xui16>
  }
}
