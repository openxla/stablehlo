// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<2x3xf32>, tensor<2x3xf32>)
    %1 = call @expected() : () -> tensor<2x6xf32>
    %2 = stablehlo.concatenate %0#0, %0#1, dim = 1 : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x6xf32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<2x6xf32>, tensor<2x6xf32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<2x3xf32>, tensor<2x3xf32>) {
    %0 = stablehlo.constant dense<[[3.43787861, 2.3870616, -0.963214695], [-0.965828657, -0.370334685, -5.5024929]]> : tensor<2x3xf32>
    %1 = stablehlo.constant dense<[[0.251219183, -0.775757908, 1.55845892], [0.488238484, 1.47575057, -0.509219706]]> : tensor<2x3xf32>
    return %0, %1 : tensor<2x3xf32>, tensor<2x3xf32>
  }
  func.func private @expected() -> tensor<2x6xf32> {
    %0 = stablehlo.constant dense<[[3.43787861, 2.3870616, -0.963214695, 0.251219183, -0.775757908, 1.55845892], [-0.965828657, -0.370334685, -5.5024929, 0.488238484, 1.47575057, -0.509219706]]> : tensor<2x6xf32>
    return %0 : tensor<2x6xf32>
  }
}
