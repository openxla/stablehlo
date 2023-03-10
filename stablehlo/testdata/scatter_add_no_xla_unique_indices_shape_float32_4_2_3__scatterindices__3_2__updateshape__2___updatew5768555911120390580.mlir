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
      %5 = stablehlo.add %arg0, %arg1 : tensor<f32>
      stablehlo.return %5 : tensor<f32>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [0, 2], scatter_dims_to_operand_dims = [0, 2]>, unique_indices = true} : (tensor<4x2x3xf32>, tensor<2xi32>, tensor<2xf32>) -> tensor<4x2x3xf32>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<4x2x3xf32>, tensor<4x2x3xf32>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x2x3xf32>, tensor<2xf32>) {
    %0 = stablehlo.constant dense<[[[-5.63943481, -1.77959085, 1.59453487], [4.08448172, 6.14536524, 0.124179229]], [[0.562016308, -4.49256516, -5.13096762], [-2.88947415, -1.01943946, 4.09113407]], [[-6.49886751, -2.05818653, -0.475863814], [4.432240e-01, 0.712762892, -7.04946375]], [[-3.6166811, -0.882766067, -2.72324681], [2.1477046, 3.12406707, -3.33359718]]]> : tensor<4x2x3xf32>
    %1 = stablehlo.constant dense<[0.792530417, -3.32832599]> : tensor<2xf32>
    return %0, %1 : tensor<4x2x3xf32>, tensor<2xf32>
  }
  func.func private @expected() -> tensor<4x2x3xf32> {
    %0 = stablehlo.constant dense<[[[-5.63943481, -1.77959085, 1.59453487], [4.08448172, 6.14536524, 0.124179229]], [[0.562016308, -4.49256516, -5.13096762], [-2.88947415, -1.01943946, 4.09113407]], [[-6.49886751, -2.05818653, -0.475863814], [4.432240e-01, 0.712762892, -7.04946375]], [[-3.6166811, -0.882766067, -1.9307164], [2.1477046, 3.12406707, -6.6619234]]]> : tensor<4x2x3xf32>
    return %0 : tensor<4x2x3xf32>
  }
}

