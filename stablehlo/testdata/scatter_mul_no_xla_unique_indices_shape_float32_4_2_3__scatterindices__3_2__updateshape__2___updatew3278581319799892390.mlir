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
      %5 = stablehlo.multiply %arg0, %arg1 : tensor<f32>
      stablehlo.return %5 : tensor<f32>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [0, 2], scatter_dims_to_operand_dims = [0, 2]>, unique_indices = true} : (tensor<4x2x3xf32>, tensor<2xi32>, tensor<2xf32>) -> tensor<4x2x3xf32>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<4x2x3xf32>, tensor<4x2x3xf32>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x2x3xf32>, tensor<2xf32>) {
    %0 = stablehlo.constant dense<[[[4.68853664, 1.64649534, 1.10540771], [-1.68123269, -0.885974347, -4.80956697]], [[-1.17188489, 2.81842232, -0.523834646], [3.09401131, -2.77528715, 1.00738275]], [[-3.55389214, 1.42628944, -2.84484315], [-1.93948877, 1.58069754, -4.26912308]], [[3.90325975, 0.585639238, -0.841291129], [-0.762754082, -0.82266587, 6.93243503]]]> : tensor<4x2x3xf32>
    %1 = stablehlo.constant dense<[-8.23026847, 0.391512454]> : tensor<2xf32>
    return %0, %1 : tensor<4x2x3xf32>, tensor<2xf32>
  }
  func.func private @expected() -> tensor<4x2x3xf32> {
    %0 = stablehlo.constant dense<[[[4.68853664, 1.64649534, 1.10540771], [-1.68123269, -0.885974347, -4.80956697]], [[-1.17188489, 2.81842232, -0.523834646], [3.09401131, -2.77528715, 1.00738275]], [[-3.55389214, 1.42628944, -2.84484315], [-1.93948877, 1.58069754, -4.26912308]], [[3.90325975, 0.585639238, 6.92405176], [-0.762754082, -0.82266587, 2.71413469]]]> : tensor<4x2x3xf32>
    return %0 : tensor<4x2x3xf32>
  }
}

