// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<5x3xf32>
    %1 = call @expected() : () -> tensor<2x0xf32>
    %2 = "stablehlo.slice"(%0) {limit_indices = dense<[3, 1]> : tensor<2xi64>, start_indices = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<5x3xf32>) -> tensor<2x0xf32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<2x0xf32>, tensor<2x0xf32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<5x3xf32> {
    %0 = stablehlo.constant dense<[[2.79533195, -1.82416129, 1.4175148], [0.707774102, 4.44844484, 5.07368326], [-1.84716344, -3.09585023, 3.31965446], [0.294154763, 2.75354218, -0.489122629], [-4.62401199, -4.18428373, 2.19709206]]> : tensor<5x3xf32>
    return %0 : tensor<5x3xf32>
  }
  func.func private @expected() -> tensor<2x0xf32> {
    %0 = stablehlo.constant dense<> : tensor<2x0xf32>
    return %0 : tensor<2x0xf32>
  }
}
