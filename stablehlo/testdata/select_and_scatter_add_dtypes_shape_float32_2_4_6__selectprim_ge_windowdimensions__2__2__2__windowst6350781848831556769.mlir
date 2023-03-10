// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<1x3x5xf32>, tensor<2x4x6xf32>)
    %1 = call @expected() : () -> tensor<2x4x6xf32>
    %2 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %3 = stablehlo.pad %0#1, %2, low = [0, 0, 0], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<2x4x6xf32>, tensor<f32>) -> tensor<2x4x6xf32>
    %4 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %5 = "stablehlo.select_and_scatter"(%3, %0#0, %4) ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %8 = stablehlo.compare  GE, %arg0, %arg1,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      stablehlo.return %8 : tensor<i1>
    }, {
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %8 = stablehlo.add %arg0, %arg1 : tensor<f32>
      stablehlo.return %8 : tensor<f32>
    }) {window_dimensions = dense<2> : tensor<3xi64>} : (tensor<2x4x6xf32>, tensor<1x3x5xf32>, tensor<f32>) -> tensor<2x4x6xf32>
    %6 = "stablehlo.slice"(%5) {limit_indices = dense<[2, 4, 6]> : tensor<3xi64>, start_indices = dense<0> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>} : (tensor<2x4x6xf32>) -> tensor<2x4x6xf32>
    %7 = stablehlo.custom_call @check.eq(%6, %1) : (tensor<2x4x6xf32>, tensor<2x4x6xf32>) -> tensor<i1>
    return %7 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<1x3x5xf32>, tensor<2x4x6xf32>) {
    %0 = stablehlo.constant dense<[[[0.653392911, 0.389078051, 6.15831279, -3.14838386, 4.23240852], [0.939077794, 0.775066137, -3.78276515, -3.55391407, 1.48504364], [-0.826507151, 1.73546553, 6.67520189, -4.91890764, 1.90688646]]]> : tensor<1x3x5xf32>
    %1 = stablehlo.constant dense<[[[-1.20494819, -1.7287128, 3.31278968, -1.64259398, -4.02583647, -4.86875248], [3.59623742, -1.65147936, 3.58527017, -2.42309833, -3.66481686, -0.292121679], [-1.1811868, 0.988534033, -7.88368415, 1.05264437, 2.48973179, -0.324023277], [-2.02255678, 2.2562077, 1.41731489, 0.438418537, -3.32704329, -3.88613129]], [[-3.61226988, -1.69340134, 2.56518149, -1.55973077, 3.38393188, 2.0728116], [-2.64545727, 0.192381918, 2.41769385, -1.59281039, 1.84527016, -4.56991529], [-0.790930211, 1.61935091, 1.55130684, 4.21068382, 1.64659691, 7.287960e-01], [-2.53947592, 1.55822825, -1.12099028, -3.9440124, -0.632119179, -3.778126]]]> : tensor<2x4x6xf32>
    return %0, %1 : tensor<1x3x5xf32>, tensor<2x4x6xf32>
  }
  func.func private @expected() -> tensor<2x4x6xf32> {
    %0 = stablehlo.constant dense<[[[0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [1.59247065, 0.000000e+00, 7.32245731, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 3.391930e+00, 0.000000e+00], [0.000000e+00, 0.908958375, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]], [[0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 1.08402467, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, -5.58038521, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]]]> : tensor<2x4x6xf32>
    return %0 : tensor<2x4x6xf32>
  }
}

