// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:3 = call @inputs() : () -> (tensor<ui32>, tensor<2x3xui32>, tensor<ui32>)
    %1 = call @expected() : () -> tensor<2x3xui32>
    %2 = stablehlo.broadcast_in_dim %0#0, dims = [] : (tensor<ui32>) -> tensor<2x3xui32>
    %3 = stablehlo.broadcast_in_dim %0#2, dims = [] : (tensor<ui32>) -> tensor<2x3xui32>
    %4 = stablehlo.clamp %2, %0#1, %3 : tensor<2x3xui32>
    %5 = stablehlo.custom_call @check.eq(%4, %1) : (tensor<2x3xui32>, tensor<2x3xui32>) -> tensor<i1>
    return %5 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<ui32>, tensor<2x3xui32>, tensor<ui32>) {
    %0 = stablehlo.constant dense<[[4, 4, 2], [1, 1, 4]]> : tensor<2x3xui32>
    %1 = stablehlo.constant dense<5> : tensor<ui32>
    %2 = stablehlo.constant dense<4> : tensor<ui32>
    return %1, %0, %2 : tensor<ui32>, tensor<2x3xui32>, tensor<ui32>
  }
  func.func private @expected() -> tensor<2x3xui32> {
    %0 = stablehlo.constant dense<4> : tensor<2x3xui32>
    return %0 : tensor<2x3xui32>
  }
}
