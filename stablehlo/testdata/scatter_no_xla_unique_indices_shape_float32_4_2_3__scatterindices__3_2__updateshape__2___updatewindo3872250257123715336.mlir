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
    %0 = stablehlo.constant dense<[[[-2.42517877, -4.12898684, -1.12015569], [-2.50819373, 3.24377537, 2.72603297]], [[3.03790736, -1.59627223, -0.783574223], [0.955120086, 1.14409721, -1.75770557]], [[0.210579351, 1.65989876, 0.784285128], [-6.03208255, 0.0774631425, 3.34217978]], [[0.777396619, 1.89477742, 1.06337619], [0.414854497, 8.08435916, 0.379786551]]]> : tensor<4x2x3xf32>
    %1 = stablehlo.constant dense<[5.7628994, 1.36500418]> : tensor<2xf32>
    return %0, %1 : tensor<4x2x3xf32>, tensor<2xf32>
  }
  func.func private @expected() -> tensor<4x2x3xf32> {
    %0 = stablehlo.constant dense<[[[-2.42517877, -4.12898684, -1.12015569], [-2.50819373, 3.24377537, 2.72603297]], [[3.03790736, -1.59627223, -0.783574223], [0.955120086, 1.14409721, -1.75770557]], [[0.210579351, 1.65989876, 0.784285128], [-6.03208255, 0.0774631425, 3.34217978]], [[0.777396619, 1.89477742, 5.7628994], [0.414854497, 8.08435916, 1.36500418]]]> : tensor<4x2x3xf32>
    return %0 : tensor<4x2x3xf32>
  }
}

