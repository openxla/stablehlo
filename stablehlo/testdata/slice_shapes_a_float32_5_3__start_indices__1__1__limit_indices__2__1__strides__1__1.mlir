// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<5x3xf32>
    %1 = call @expected() : () -> tensor<1x0xf32>
    %2 = "stablehlo.slice"(%0) {limit_indices = dense<[2, 1]> : tensor<2xi64>, start_indices = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<5x3xf32>) -> tensor<1x0xf32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<1x0xf32>, tensor<1x0xf32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<5x3xf32> {
    %0 = stablehlo.constant dense<[[-3.36093402, -0.727180302, -1.36010623], [-1.01435328, 0.688827634, -0.296935827], [-2.48586321, -1.59691119, -2.82093692], [1.89409471, 4.85353708, 2.03780484], [2.1872561, 0.797380387, 1.21235549]]> : tensor<5x3xf32>
    return %0 : tensor<5x3xf32>
  }
  func.func private @expected() -> tensor<1x0xf32> {
    %0 = stablehlo.constant dense<> : tensor<1x0xf32>
    return %0 : tensor<1x0xf32>
  }
}
