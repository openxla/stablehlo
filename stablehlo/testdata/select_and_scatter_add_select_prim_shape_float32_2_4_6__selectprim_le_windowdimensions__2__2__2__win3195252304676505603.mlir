// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<1x3x5xf32>, tensor<2x4x6xf32>)
    %1 = call @expected() : () -> tensor<2x4x6xf32>
    %2 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %3 = stablehlo.pad %0#1, %2, low = [0, 0, 0], high = [0, 0, 0], interior = [0, 0, 0] : (tensor<2x4x6xf32>, tensor<f32>) -> tensor<2x4x6xf32>
    %4 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %5 = "stablehlo.select_and_scatter"(%3, %0#0, %4) ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %8 = stablehlo.compare  LE, %arg0, %arg1,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
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
    %0 = stablehlo.constant dense<[[[-3.5882771, 2.35350013, 1.55692399, 2.61422753, -2.5195365], [-1.48691666, 2.89030099, -0.248438463, 1.79996824, 2.0120194], [-6.14380312, -8.12209892, -1.95739663, 1.39226353, -2.93439102]]]> : tensor<1x3x5xf32>
    %1 = stablehlo.constant dense<[[[4.710820e-02, -2.94462276, -2.53436804, 1.79387867, -1.6815784, -4.2265892], [-0.308423966, -3.41425824, 5.84160852, -1.74297357, -5.21533585, -1.72261572], [2.25816393, 4.39420319, -3.3486557, 2.63009882, -1.27760208, -4.73301697], [-4.28886318, -3.18570757, 1.82154644, 4.95820951, 1.19878316, 1.53634322]], [[-1.13555777, 2.69783378, 0.697399139, 6.01423073, -5.01759434, 4.10634899], [-0.0844627768, 3.53833771, 0.499059916, -2.03713226, 0.759291887, -3.36340928], [2.99461055, -0.92390871, 5.75102282, 0.537356377, -3.61243868, -1.20229185], [2.21204185, -3.59621668, -5.28717518, 3.35203385, 1.37710178, -1.83568382]]]> : tensor<2x4x6xf32>
    return %0, %1 : tensor<1x3x5xf32>, tensor<2x4x6xf32>
  }
  func.func private @expected() -> tensor<2x4x6xf32> {
    %0 = stablehlo.constant dense<[[[0.000000e+00, 0.000000e+00, 1.55692399, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.168607473, 0.000000e+00, 0.000000e+00, 3.90667868, 0.000000e+00], [0.000000e+00, 0.000000e+00, -0.248438463, 0.000000e+00, 0.000000e+00, -2.93439102], [-6.14380312, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]], [[0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 1.39226353, 0.000000e+00], [0.000000e+00, 0.000000e+00, -10.0794954, 0.000000e+00, 0.000000e+00, 0.000000e+00]]]> : tensor<2x4x6xf32>
    return %0 : tensor<2x4x6xf32>
  }
}

