// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<2x3xi1>
    %1 = call @expected() : () -> tensor<3xi1>
    %2 = stablehlo.constant dense<true> : tensor<i1>
    %3 = stablehlo.reduce(%0 init: %2) across dimensions = [0] : (tensor<2x3xi1>, tensor<i1>) -> tensor<3xi1>
     reducer(%arg0: tensor<i1>, %arg1: tensor<i1>)  {
      %5 = stablehlo.and %arg0, %arg1 : tensor<i1>
      stablehlo.return %5 : tensor<i1>
    }
    %4 = stablehlo.custom_call @check.eq(%3, %1) : (tensor<3xi1>, tensor<3xi1>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> tensor<2x3xi1> {
    %0 = stablehlo.constant dense<true> : tensor<2x3xi1>
    return %0 : tensor<2x3xi1>
  }
  func.func private @expected() -> tensor<3xi1> {
    %0 = stablehlo.constant dense<true> : tensor<3xi1>
    return %0 : tensor<3xi1>
  }
}
