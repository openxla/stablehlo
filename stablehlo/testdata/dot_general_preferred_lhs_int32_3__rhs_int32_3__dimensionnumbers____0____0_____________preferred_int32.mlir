// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<3xi32>, tensor<3xi32>)
    %1 = call @expected() : () -> tensor<i32>
    %2 = "stablehlo.dot_general"(%0#0, %0#1) {dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [0], rhs_contracting_dimensions = [0]>} : (tensor<3xi32>, tensor<3xi32>) -> tensor<i32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<i32>, tensor<i32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<3xi32>, tensor<3xi32>) {
    %0 = stablehlo.constant dense<[3, 4, 0]> : tensor<3xi32>
    %1 = stablehlo.constant dense<[0, -2, -2]> : tensor<3xi32>
    return %0, %1 : tensor<3xi32>, tensor<3xi32>
  }
  func.func private @expected() -> tensor<i32> {
    %0 = stablehlo.constant dense<-8> : tensor<i32>
    return %0 : tensor<i32>
  }
}

