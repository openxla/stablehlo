// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<2x3xi1>
    %1 = call @expected() : () -> tensor<2x3xi1>
    %2 = stablehlo.bitcast_convert %0 : (tensor<2x3xi1>) -> tensor<2x3xi1>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<2x3xi1>, tensor<2x3xi1>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<2x3xi1> {
    %0 = stablehlo.constant dense<true> : tensor<2x3xi1>
    return %0 : tensor<2x3xi1>
  }
  func.func private @expected() -> tensor<2x3xi1> {
    %0 = stablehlo.constant dense<true> : tensor<2x3xi1>
    return %0 : tensor<2x3xi1>
  }
}
