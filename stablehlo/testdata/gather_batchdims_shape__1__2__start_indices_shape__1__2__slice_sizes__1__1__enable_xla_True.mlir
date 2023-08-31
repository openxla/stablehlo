// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<1x2xf32>, tensor<1x2xi32>)
    %1 = call @expected() : () -> tensor<1xf32>
    %2 = "stablehlo.gather"(%0#0, %0#1) {dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>, slice_sizes = dense<1> : tensor<2xi64>} : (tensor<1x2xf32>, tensor<1x2xi32>) -> tensor<1xf32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<1xf32>, tensor<1xf32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<1x2xf32>, tensor<1x2xi32>) {
    %0 = stablehlo.constant dense<[[-2.72349977, -0.208018199]]> : tensor<1x2xf32>
    %1 = stablehlo.constant dense<[[0, 1]]> : tensor<1x2xi32>
    return %0, %1 : tensor<1x2xf32>, tensor<1x2xi32>
  }
  func.func private @expected() -> tensor<1xf32> {
    %0 = stablehlo.constant dense<-0.208018199> : tensor<1xf32>
    return %0 : tensor<1xf32>
  }
}

