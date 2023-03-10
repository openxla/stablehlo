// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<[3, 2]> : tensor<2xi32>
    %1:2 = call @inputs() : () -> (tensor<4x2x3xf32>, tensor<2xf32>)
    %2 = call @expected() : () -> tensor<4x2x3xf32>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %5 = stablehlo.minimum %arg0, %arg1 : tensor<f32>
      stablehlo.return %5 : tensor<f32>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [0, 2], scatter_dims_to_operand_dims = [0, 2]>, unique_indices = true} : (tensor<4x2x3xf32>, tensor<2xi32>, tensor<2xf32>) -> tensor<4x2x3xf32>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<4x2x3xf32>, tensor<4x2x3xf32>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x2x3xf32>, tensor<2xf32>) {
    %0 = stablehlo.constant dense<[[[-5.98844194, -7.21278667, 4.22167397], [2.10766792, -1.18228841, 1.73586547]], [[1.62650871, -3.37734795, 1.07885122], [0.950231134, 1.93563771, -4.28763914]], [[-5.74428797, -1.1279633, -1.44278562], [-2.76901436, 1.01394463, 2.47923112]], [[5.4248147, -1.17900741, 0.712611198], [1.24946117, -6.13739538, 0.774065196]]]> : tensor<4x2x3xf32>
    %1 = stablehlo.constant dense<[-0.135103092, 0.32835564]> : tensor<2xf32>
    return %0, %1 : tensor<4x2x3xf32>, tensor<2xf32>
  }
  func.func private @expected() -> tensor<4x2x3xf32> {
    %0 = stablehlo.constant dense<[[[-5.98844194, -7.21278667, 4.22167397], [2.10766792, -1.18228841, 1.73586547]], [[1.62650871, -3.37734795, 1.07885122], [0.950231134, 1.93563771, -4.28763914]], [[-5.74428797, -1.1279633, -1.44278562], [-2.76901436, 1.01394463, 2.47923112]], [[5.4248147, -1.17900741, -0.135103092], [1.24946117, -6.13739538, 0.32835564]]]> : tensor<4x2x3xf32>
    return %0 : tensor<4x2x3xf32>
  }
}

