// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<7xf32>
    %1 = call @expected() : () -> tensor<3xf32>
    %2 = "stablehlo.slice"(%0) {limit_indices = array<i64: 7>, start_indices = array<i64: 4>, strides = array<i64: 1>} : (tensor<7xf32>) -> tensor<3xf32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<3xf32>, tensor<3xf32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<7xf32> {
    %0 = stablehlo.constant dense<[-2.02720737, -6.03291559, 2.70835233, 1.80772364, 4.24452925, 0.507206202, 5.57621193]> : tensor<7xf32>
    return %0 : tensor<7xf32>
  }
  func.func private @expected() -> tensor<3xf32> {
    %0 = stablehlo.constant dense<[4.24452925, 0.507206202, 5.57621193]> : tensor<3xf32>
    return %0 : tensor<3xf32>
  }
}
