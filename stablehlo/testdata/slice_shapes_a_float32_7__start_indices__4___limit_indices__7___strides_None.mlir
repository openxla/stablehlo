// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<7xf32>
    %1 = call @expected() : () -> tensor<3xf32>
    %2 = "stablehlo.slice"(%0) {limit_indices = dense<7> : tensor<1xi64>, start_indices = dense<4> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<7xf32>) -> tensor<3xf32>
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
