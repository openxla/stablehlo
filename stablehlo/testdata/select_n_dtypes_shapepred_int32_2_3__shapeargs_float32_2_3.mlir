// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:4 = call @inputs() : () -> (tensor<2x3xi32>, tensor<2x3xf32>, tensor<2x3xf32>, tensor<2x3xf32>)
    %1 = call @expected() : () -> tensor<2x3xf32>
    %2 = stablehlo.constant dense<1> : tensor<i32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<i32>) -> tensor<2x3xi32>
    %4 = stablehlo.compare  LT, %0#0, %3,  SIGNED : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi1>
    %5 = stablehlo.constant dense<2> : tensor<i32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [] : (tensor<i32>) -> tensor<2x3xi32>
    %7 = stablehlo.compare  LT, %0#0, %6,  SIGNED : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi1>
    %8 = stablehlo.select %7, %0#2, %0#3 : tensor<2x3xi1>, tensor<2x3xf32>
    %9 = stablehlo.select %4, %0#1, %8 : tensor<2x3xi1>, tensor<2x3xf32>
    %10 = stablehlo.custom_call @check.eq(%9, %1) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<i1>
    return %10 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<2x3xi32>, tensor<2x3xf32>, tensor<2x3xf32>, tensor<2x3xf32>) {
    %0 = stablehlo.constant dense<[[1, 0, 0], [0, 1, 2]]> : tensor<2x3xi32>
    %1 = stablehlo.constant dense<[[1.53065205, -1.85948396, -0.36665079], [-6.33873701, 1.70852697, -3.0464015]]> : tensor<2x3xf32>
    %2 = stablehlo.constant dense<[[-1.3053329, -3.28220105, 2.17391253], [5.61426926, -1.7552247, -0.547213137]]> : tensor<2x3xf32>
    %3 = stablehlo.constant dense<[[4.27107477, -4.0092535, 0.189415693], [1.6731391, -1.34621441, 3.26871157]]> : tensor<2x3xf32>
    return %0, %1, %2, %3 : tensor<2x3xi32>, tensor<2x3xf32>, tensor<2x3xf32>, tensor<2x3xf32>
  }
  func.func private @expected() -> tensor<2x3xf32> {
    %0 = stablehlo.constant dense<[[-1.3053329, -1.85948396, -0.36665079], [-6.33873701, -1.7552247, 3.26871157]]> : tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
  }
}
