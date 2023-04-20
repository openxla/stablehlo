// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<1x3x4xf32>, tensor<1x4x3xf32>)
    %1 = call @expected() : () -> tensor<1xf32>
    %2 = "stablehlo.dot_general"(%0#0, %0#1) {dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2, 1], rhs_contracting_dimensions = [1, 2]>, precision_config = [#stablehlo<precision HIGHEST>, #stablehlo<precision HIGHEST>]} : (tensor<1x3x4xf32>, tensor<1x4x3xf32>) -> tensor<1xf32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<1xf32>, tensor<1xf32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<1x3x4xf32>, tensor<1x4x3xf32>) {
    %0 = stablehlo.constant dense<[[[2.59234142, -2.35737705, 3.07461166, -5.77705336], [-3.64460349, 0.689637601, -0.0876942202, 1.62593222], [-4.65739489, 0.247004092, -1.08101177, -0.238710642]]]> : tensor<1x3x4xf32>
    %1 = stablehlo.constant dense<[[[-3.64053726, 1.4618907, -0.867068588], [2.86438012, -2.70172548, -0.4580172], [0.140140817, -0.666462898, -7.101100e-01], [1.02142382, 0.236523077, 0.760420739]]]> : tensor<1x4x3xf32>
    return %0, %1 : tensor<1x3x4xf32>, tensor<1x4x3xf32>
  }
  func.func private @expected() -> tensor<1xf32> {
    %0 = stablehlo.constant dense<-23.896822> : tensor<1xf32>
    return %0 : tensor<1xf32>
  }
}
