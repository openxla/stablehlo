// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<5xf32>
    %1 = call @expected() : () -> tensor<2xf32>
    %2 = "stablehlo.slice"(%0) {limit_indices = dense<5> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<2> : tensor<1xi64>} : (tensor<5xf32>) -> tensor<2xf32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<2xf32>, tensor<2xf32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<5xf32> {
    %0 = stablehlo.constant dense<[6.3510623, -0.349791348, 1.07969737, 6.3544569, 3.81555367]> : tensor<5xf32>
    return %0 : tensor<5xf32>
  }
  func.func private @expected() -> tensor<2xf32> {
    %0 = stablehlo.constant dense<[-0.349791348, 6.3544569]> : tensor<2xf32>
    return %0 : tensor<2xf32>
  }
}
