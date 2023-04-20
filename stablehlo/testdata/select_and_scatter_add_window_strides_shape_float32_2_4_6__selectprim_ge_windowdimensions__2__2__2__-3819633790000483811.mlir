// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<1x2x2xf32>, tensor<2x4x6xf32>)
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
    }) {window_dimensions = dense<2> : tensor<3xi64>, window_strides = dense<[1, 2, 3]> : tensor<3xi64>} : (tensor<2x4x6xf32>, tensor<1x2x2xf32>, tensor<f32>) -> tensor<2x4x6xf32>
    %6 = "stablehlo.slice"(%5) {limit_indices = dense<[2, 4, 6]> : tensor<3xi64>, start_indices = dense<0> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>} : (tensor<2x4x6xf32>) -> tensor<2x4x6xf32>
    %7 = stablehlo.custom_call @check.eq(%6, %1) : (tensor<2x4x6xf32>, tensor<2x4x6xf32>) -> tensor<i1>
    return %7 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<1x2x2xf32>, tensor<2x4x6xf32>) {
    %0 = stablehlo.constant dense<[[[0.71100831, 3.48205233], [0.894208729, -4.17481089]]]> : tensor<1x2x2xf32>
    %1 = stablehlo.constant dense<[[[-0.644808292, 1.92047977, -1.91922772, 0.999633669, -3.48618364, -6.41643095], [2.23947692, 1.80902565, -4.08624411, 7.93703556, -1.76496863, -0.806669354], [-3.46046758, -2.34411597, 1.2211386, 0.969038367, 0.487834394, 6.458230e+00], [2.1669035, -9.5307064, 1.00031853, 2.82388973, -8.67260932, -1.72873378]], [[1.03890836, -2.66181707, -0.909946203, -6.5063858, -4.31322432, -3.11935401], [2.90081286, 0.280726552, -2.574870e+00, -1.99189842, 1.59743047, -0.313812256], [0.814279437, 3.82442188, -0.556298196, -0.34467873, 0.913265347, -6.68484449], [-2.49808025, 4.06428099, -0.112926006, 0.382048458, 1.47863889, 3.5611577]]]> : tensor<2x4x6xf32>
    return %0, %1 : tensor<1x2x2xf32>, tensor<2x4x6xf32>
  }
  func.func private @expected() -> tensor<2x4x6xf32> {
    %0 = stablehlo.constant dense<[[[0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 3.48205233, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, -4.17481089, 0.000000e+00, 0.000000e+00]], [[0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.71100831, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.894208729, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]]]> : tensor<2x4x6xf32>
    return %0 : tensor<2x4x6xf32>
  }
}

