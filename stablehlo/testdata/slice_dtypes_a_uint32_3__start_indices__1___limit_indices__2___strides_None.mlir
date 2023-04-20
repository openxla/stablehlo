// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<3xui32>
    %1 = call @expected() : () -> tensor<1xui32>
    %2 = "stablehlo.slice"(%0) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<3xui32>) -> tensor<1xui32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<1xui32>, tensor<1xui32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<3xui32> {
    %0 = stablehlo.constant dense<[1, 0, 3]> : tensor<3xui32>
    return %0 : tensor<3xui32>
  }
  func.func private @expected() -> tensor<1xui32> {
    %0 = stablehlo.constant dense<0> : tensor<1xui32>
    return %0 : tensor<1xui32>
  }
}
