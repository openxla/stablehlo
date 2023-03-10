// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<8xf32>
    %1 = call @expected() : () -> tensor<3xf32>
    %2 = "stablehlo.slice"(%0) {limit_indices = dense<6> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<2> : tensor<1xi64>} : (tensor<8xf32>) -> tensor<3xf32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<3xf32>, tensor<3xf32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<8xf32> {
    %0 = stablehlo.constant dense<[1.30773413, 0.875493407, 0.384833664, 4.16633749, -0.0197476614, -5.21694708, -2.27677536, 4.62631655]> : tensor<8xf32>
    return %0 : tensor<8xf32>
  }
  func.func private @expected() -> tensor<3xf32> {
    %0 = stablehlo.constant dense<[0.875493407, 4.16633749, -5.21694708]> : tensor<3xf32>
    return %0 : tensor<3xf32>
  }
}
