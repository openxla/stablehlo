// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<3x3xf32>, tensor<3x3xf32>)
    %1 = call @expected() : () -> tensor<3x3xf32>
    %2 = stablehlo.maximum %0#0, %0#1 : tensor<3x3xf32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<3x3xf32>, tensor<3x3xf32>) {
    %0 = stablehlo.constant dense<[[0x7FC00000, 0x7FC00000, 0x7FC00000], [0x7F800000, 0x7F800000, 0x7F800000], [0xFF800000, 0xFF800000, 0xFF800000]]> : tensor<3x3xf32>
    %1 = stablehlo.constant dense<[[0x7FC00000, 0x7F800000, 0xFF800000], [0x7FC00000, 0x7F800000, 0xFF800000], [0x7FC00000, 0x7F800000, 0xFF800000]]> : tensor<3x3xf32>
    return %0, %1 : tensor<3x3xf32>, tensor<3x3xf32>
  }
  func.func private @expected() -> tensor<3x3xf32> {
    %0 = stablehlo.constant dense<[[0x7FC00000, 0x7FC00000, 0x7FC00000], [0x7FC00000, 0x7F800000, 0x7F800000], [0x7FC00000, 0x7F800000, 0xFF800000]]> : tensor<3x3xf32>
    return %0 : tensor<3x3xf32>
  }
}
