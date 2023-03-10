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
      stablehlo.return %arg1 : tensor<f32>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [0, 2], scatter_dims_to_operand_dims = [0, 2]>, unique_indices = true} : (tensor<4x2x3xf32>, tensor<2xi32>, tensor<2xf32>) -> tensor<4x2x3xf32>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<4x2x3xf32>, tensor<4x2x3xf32>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x2x3xf32>, tensor<2xf32>) {
    %0 = stablehlo.constant dense<[[[6.0793004, 1.53058326, -0.960142791], [3.53896618, -0.37528339, 1.52082372]], [[-5.9812665, 1.88195837, -0.361230522], [1.87618184, -2.87599587, -1.42694318]], [[-1.9547224, 2.28265905, 1.45795095], [3.95608616, 3.56203961, 1.72593415]], [[0.502125442, 2.17254972, 2.90101123], [-0.960227131, -2.10696721, -2.77633953]]]> : tensor<4x2x3xf32>
    %1 = stablehlo.constant dense<[-3.24349618, 1.97058392]> : tensor<2xf32>
    return %0, %1 : tensor<4x2x3xf32>, tensor<2xf32>
  }
  func.func private @expected() -> tensor<4x2x3xf32> {
    %0 = stablehlo.constant dense<[[[6.0793004, 1.53058326, -0.960142791], [3.53896618, -0.37528339, 1.52082372]], [[-5.9812665, 1.88195837, -0.361230522], [1.87618184, -2.87599587, -1.42694318]], [[-1.9547224, 2.28265905, 1.45795095], [3.95608616, 3.56203961, 1.72593415]], [[0.502125442, 2.17254972, -3.24349618], [-0.960227131, -2.10696721, 1.97058392]]]> : tensor<4x2x3xf32>
    return %0 : tensor<4x2x3xf32>
  }
}

