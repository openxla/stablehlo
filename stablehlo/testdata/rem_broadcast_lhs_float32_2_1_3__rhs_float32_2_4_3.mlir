// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<2x1x3xf32>, tensor<2x4x3xf32>)
    %1 = call @expected() : () -> tensor<2x4x3xf32>
    %2 = stablehlo.broadcast_in_dim %0#0, dims = [0, 1, 2] : (tensor<2x1x3xf32>) -> tensor<2x4x3xf32>
    %3 = stablehlo.remainder %2, %0#1 : tensor<2x4x3xf32>
    %4 = stablehlo.custom_call @check.eq(%3, %1) : (tensor<2x4x3xf32>, tensor<2x4x3xf32>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<2x1x3xf32>, tensor<2x4x3xf32>) {
    %0 = stablehlo.constant dense<[[[0.932311177, -6.13186073, 0.688831508]], [[1.70496666, -1.60266161, -2.7677772]]]> : tensor<2x1x3xf32>
    %1 = stablehlo.constant dense<[[[-1.17527223, 3.92543602, -2.16753602], [-2.07166934, -3.55415416, -0.390092105], [3.5661242, 1.547020e+00, 0.103896178], [-2.20307732, -1.3601681, -2.279356]], [[-0.526593149, -0.184299067, -5.13481617], [-2.73182583, -0.481056184, 3.8781662], [-2.96626377, -4.69370842, 1.76389694], [-1.98348045, 0.0347406194, -4.51008463]]]> : tensor<2x4x3xf32>
    return %0, %1 : tensor<2x1x3xf32>, tensor<2x4x3xf32>
  }
  func.func private @expected() -> tensor<2x4x3xf32> {
    %0 = stablehlo.constant dense<[[[0.932311177, -2.20642471, 0.688831508], [0.932311177, -2.57770658, 0.298739403], [0.932311177, -1.49080086, 0.0654544383], [0.932311177, -0.691188335, 0.688831508]], [[0.125187218, -0.128269076, -2.7677772], [1.70496666, -0.159493059, -2.7677772], [1.70496666, -1.60266161, -1.00388026], [1.70496666, -0.00459311903, -2.7677772]]]> : tensor<2x4x3xf32>
    return %0 : tensor<2x4x3xf32>
  }
}
