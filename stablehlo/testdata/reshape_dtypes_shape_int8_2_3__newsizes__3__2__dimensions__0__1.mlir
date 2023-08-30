// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<2x3xi8>
    %1 = call @expected() : () -> tensor<3x2xi8>
    %2 = stablehlo.reshape %0 : (tensor<2x3xi8>) -> tensor<3x2xi8>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<3x2xi8>, tensor<3x2xi8>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<2x3xi8> {
    %0 = stablehlo.constant dense<[[6, 0, -1], [3, 6, 3]]> : tensor<2x3xi8>
    return %0 : tensor<2x3xi8>
  }
  func.func private @expected() -> tensor<3x2xi8> {
    %0 = stablehlo.constant dense<[[6, 0], [-1, 3], [6, 3]]> : tensor<3x2xi8>
    return %0 : tensor<3x2xi8>
  }
}
