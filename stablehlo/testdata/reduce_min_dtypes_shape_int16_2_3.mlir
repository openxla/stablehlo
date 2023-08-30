// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<2x3xi16>
    %1 = call @expected() : () -> tensor<3xi16>
    %2 = stablehlo.constant dense<32767> : tensor<i16>
    %3 = stablehlo.reduce(%0 init: %2) across dimensions = [0] : (tensor<2x3xi16>, tensor<i16>) -> tensor<3xi16>
     reducer(%arg0: tensor<i16>, %arg1: tensor<i16>)  {
      %5 = stablehlo.minimum %arg0, %arg1 : tensor<i16>
      stablehlo.return %5 : tensor<i16>
    }
    %4 = stablehlo.custom_call @check.eq(%3, %1) : (tensor<3xi16>, tensor<3xi16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> tensor<2x3xi16> {
    %0 = stablehlo.constant dense<[[-1, 1, 2], [3, 0, 0]]> : tensor<2x3xi16>
    return %0 : tensor<2x3xi16>
  }
  func.func private @expected() -> tensor<3xi16> {
    %0 = stablehlo.constant dense<[-1, 0, 0]> : tensor<3xi16>
    return %0 : tensor<3xi16>
  }
}
