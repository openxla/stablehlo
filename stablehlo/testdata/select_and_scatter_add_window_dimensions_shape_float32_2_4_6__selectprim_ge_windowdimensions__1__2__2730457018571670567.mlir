// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<2x3x4xf32>, tensor<2x4x6xf32>)
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
    }) {window_dimensions = dense<[1, 2, 3]> : tensor<3xi64>} : (tensor<2x4x6xf32>, tensor<2x3x4xf32>, tensor<f32>) -> tensor<2x4x6xf32>
    %6 = "stablehlo.slice"(%5) {limit_indices = dense<[2, 4, 6]> : tensor<3xi64>, start_indices = dense<0> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>} : (tensor<2x4x6xf32>) -> tensor<2x4x6xf32>
    %7 = stablehlo.custom_call @check.eq(%6, %1) : (tensor<2x4x6xf32>, tensor<2x4x6xf32>) -> tensor<i1>
    return %7 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<2x3x4xf32>, tensor<2x4x6xf32>) {
    %0 = stablehlo.constant dense<[[[-3.48830891, -3.19353199, 2.28055644, 2.2177515], [0.0118685663, 0.384935558, 0.0436423272, 1.72155821], [-0.294638127, 1.2129606, 2.43438959, -2.94057393]], [[-0.327378184, 6.76751661, 2.73681092, -0.780638754], [2.2331965, 0.044658348, -2.42868066, 5.36513758], [4.87777662, -1.61283994, 4.322649, -1.46327734]]]> : tensor<2x3x4xf32>
    %1 = stablehlo.constant dense<[[[5.25141239, 3.87783408, -0.991326749, 6.11796092, 4.492390e+00, 1.9618969], [3.46811748, -3.58463693, 2.69827414, -1.74762201, 1.9004128, 3.31908894], [-0.00722294161, 1.70056069, 3.44013405, -1.7048043, -0.0448316224, -0.483042687], [1.49379408, -2.06683135, -0.0542361811, -1.04858434, -1.42931855, -2.4098866]], [[2.11631703, 2.95481014, 3.0704422, 2.12525415, 2.58434534, 1.856870e+00], [4.08923101, 4.79104424, -2.03120923, -3.18497276, -3.18707657, -2.59061146], [1.97100568, -3.32964778, 0.107630759, -4.2467742, -1.791152, 0.917880535], [-2.2280612, 1.6504277, -1.59917331, 1.18973446, -1.74773097, -0.222520605]]]> : tensor<2x4x6xf32>
    return %0, %1 : tensor<2x3x4xf32>, tensor<2x4x6xf32>
  }
  func.func private @expected() -> tensor<2x4x6xf32> {
    %0 = stablehlo.constant dense<[[[-3.48830891, 0.000000e+00, 0.000000e+00, 1.30477595, 0.000000e+00, 0.000000e+00], [0.0118685663, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 1.72155821], [0.000000e+00, 0.000000e+00, 3.781290e+00, 0.000000e+00, -2.94057393, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]], [[0.000000e+00, 0.000000e+00, 2.73681092, 0.000000e+00, -0.780638754, 0.000000e+00], [0.000000e+00, 8.71799373, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [4.87777662, 0.000000e+00, -2.42868066, 0.000000e+00, 0.000000e+00, 5.36513758], [0.000000e+00, -1.61283994, 0.000000e+00, 2.85937166, 0.000000e+00, 0.000000e+00]]]> : tensor<2x4x6xf32>
    return %0 : tensor<2x4x6xf32>
  }
}

