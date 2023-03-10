// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<4x6xf32>
    %1 = call @expected() : () -> tensor<3x5xf32>
    %2 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f32>) -> tensor<f32>
    %4 = "stablehlo.reduce_window"(%0, %3) ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %6 = stablehlo.minimum %arg0, %arg1 : tensor<f32>
      stablehlo.return %6 : tensor<f32>
    }) {window_dimensions = dense<2> : tensor<2xi64>} : (tensor<4x6xf32>, tensor<f32>) -> tensor<3x5xf32>
    %5 = stablehlo.custom_call @check.eq(%4, %1) : (tensor<3x5xf32>, tensor<3x5xf32>) -> tensor<i1>
    return %5 : tensor<i1>
  }
  func.func private @inputs() -> tensor<4x6xf32> {
    %0 = stablehlo.constant dense<[[-0.483963817, -0.622043848, -0.104581401, 4.05938482, 0.623623967, 0.0347911902], [-3.40358257, -7.44877577, 1.34049273, -1.30082273, -1.15715182, -0.77442491], [2.06066585, 0.493570179, 7.3717289, 2.15436482, 2.13179827, 1.56632078], [1.62059426, 1.58352363, -3.50825357, -4.6696353, -2.61498523, 1.51394856]]> : tensor<4x6xf32>
    return %0 : tensor<4x6xf32>
  }
  func.func private @expected() -> tensor<3x5xf32> {
    %0 = stablehlo.constant dense<[[-7.44877577, -7.44877577, -1.30082273, -1.30082273, -1.15715182], [-7.44877577, -7.44877577, -1.30082273, -1.30082273, -1.15715182], [0.493570179, -3.50825357, -4.6696353, -4.6696353, -2.61498523]]> : tensor<3x5xf32>
    return %0 : tensor<3x5xf32>
  }
}

