// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<4x6xf32>
    %1 = call @expected() : () -> tensor<3x4xf32>
    %2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f32>) -> tensor<f32>
    %4 = "stablehlo.reduce_window"(%0, %3) ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %6 = stablehlo.add %arg0, %arg1 : tensor<f32>
      stablehlo.return %6 : tensor<f32>
    }) {window_dilations = dense<[1, 2]> : tensor<2xi64>, window_dimensions = dense<2> : tensor<2xi64>} : (tensor<4x6xf32>, tensor<f32>) -> tensor<3x4xf32>
    %5 = stablehlo.custom_call @check.eq(%4, %1) : (tensor<3x4xf32>, tensor<3x4xf32>) -> tensor<i1>
    return %5 : tensor<i1>
  }
  func.func private @inputs() -> tensor<4x6xf32> {
    %0 = stablehlo.constant dense<[[-0.349086285, 2.29166055, 2.77946019, 0.926069736, -2.0010767, 2.91728592], [-1.0548476, 0.558676183, -1.40546167, 3.278210e+00, 2.32439208, -5.22059536], [-1.90627372, -1.63722074, -2.64005661, -4.91473436, -0.425251544, -2.00524092], [4.58750486, 1.19172537, 1.32113636, -1.53597248, -3.23914957, -0.391465425]]> : tensor<4x6xf32>
    return %0 : tensor<4x6xf32>
  }
  func.func private @expected() -> tensor<3x4xf32> {
    %0 = stablehlo.constant dense<[[-0.02993536, 7.05461645, 1.6973139, 1.90097046], [-7.00663948, -2.71506882, -2.1463778, -8.862360e+00], [1.36231077, -6.89620256, -4.98332119, -8.84741306]]> : tensor<3x4xf32>
    return %0 : tensor<3x4xf32>
  }
}

