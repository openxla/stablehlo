// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<1x4x3x2x1xf32>
    %1 = call @expected() : () -> tensor<1x3x2x1x1xf32>
    %2 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f32>) -> tensor<f32>
    %4 = "stablehlo.reduce_window"(%0, %3) ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %6 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
      stablehlo.return %6 : tensor<f32>
    }) {window_dimensions = dense<[1, 2, 2, 2, 1]> : tensor<5xi64>} : (tensor<1x4x3x2x1xf32>, tensor<f32>) -> tensor<1x3x2x1x1xf32>
    %5 = stablehlo.custom_call @check.eq(%4, %1) : (tensor<1x3x2x1x1xf32>, tensor<1x3x2x1x1xf32>) -> tensor<i1>
    return %5 : tensor<i1>
  }
  func.func private @inputs() -> tensor<1x4x3x2x1xf32> {
    %0 = stablehlo.constant dense<[[[[[-2.32010245], [-2.93072677]], [[-0.483071774], [2.6272738]], [[-3.52906466], [2.43745828]]], [[[-8.67481136], [-4.36985922]], [[-0.180730611], [3.9129684]], [[2.23849201], [-0.0599784292]]], [[[-3.81792307], [-0.396719128]], [[2.38807416], [-3.4222374]], [[-3.99882317], [1.77649593]]], [[[-6.771070e-01], [-2.96301198]], [[1.40673661], [5.02723694]], [[1.16101742], [1.81690824]]]]]> : tensor<1x4x3x2x1xf32>
    return %0 : tensor<1x4x3x2x1xf32>
  }
  func.func private @expected() -> tensor<1x3x2x1x1xf32> {
    %0 = stablehlo.constant dense<[[[[[3.9129684]], [[3.9129684]]], [[[3.9129684]], [[3.9129684]]], [[[5.02723694]], [[5.02723694]]]]]> : tensor<1x3x2x1x1xf32>
    return %0 : tensor<1x3x2x1x1xf32>
  }
}

