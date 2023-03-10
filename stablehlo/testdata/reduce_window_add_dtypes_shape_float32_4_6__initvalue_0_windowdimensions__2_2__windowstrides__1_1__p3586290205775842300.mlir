// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<4x6xf32>
    %1 = call @expected() : () -> tensor<3x5xf32>
    %2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f32>) -> tensor<f32>
    %4 = "stablehlo.reduce_window"(%0, %3) ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %6 = stablehlo.add %arg0, %arg1 : tensor<f32>
      stablehlo.return %6 : tensor<f32>
    }) {window_dimensions = dense<2> : tensor<2xi64>} : (tensor<4x6xf32>, tensor<f32>) -> tensor<3x5xf32>
    %5 = stablehlo.custom_call @check.eq(%4, %1) : (tensor<3x5xf32>, tensor<3x5xf32>) -> tensor<i1>
    return %5 : tensor<i1>
  }
  func.func private @inputs() -> tensor<4x6xf32> {
    %0 = stablehlo.constant dense<[[-0.302271217, 1.56679356, -2.34925342, 3.33000422, 3.45655012, -3.36118746], [0.762212515, -3.36874557, 0.896323144, 1.07749403, -1.8407656, -3.77434158], [1.44146609, 0.537757099, -2.36054492, -1.15650773, 1.57006264, 0.400099605], [-3.1880126, -2.44533229, -0.41297102, 4.15435505, -0.021739006, 0.54186815]]> : tensor<4x6xf32>
    return %0 : tensor<4x6xf32>
  }
  func.func private @expected() -> tensor<3x5xf32> {
    %0 = stablehlo.constant dense<[[-1.34201074, -3.25488234, 2.95456791, 6.023283, -5.5197444], [-0.627309858, -4.295210e+00, -1.54323554, -0.349716663, -3.64494491], [-3.65412164, -4.68109083, 0.224331379, 4.54617119, 2.49029136]]> : tensor<3x5xf32>
    return %0 : tensor<3x5xf32>
  }
}

