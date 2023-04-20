// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<4x6xf32>
    %1 = call @expected() : () -> tensor<3x10xf32>
    %2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f32>) -> tensor<f32>
    %4 = "stablehlo.reduce_window"(%0, %3) ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %6 = stablehlo.add %arg0, %arg1 : tensor<f32>
      stablehlo.return %6 : tensor<f32>
    }) {base_dilations = dense<[1, 2]> : tensor<2xi64>, window_dimensions = dense<2> : tensor<2xi64>} : (tensor<4x6xf32>, tensor<f32>) -> tensor<3x10xf32>
    %5 = stablehlo.custom_call @check.eq(%4, %1) : (tensor<3x10xf32>, tensor<3x10xf32>) -> tensor<i1>
    return %5 : tensor<i1>
  }
  func.func private @inputs() -> tensor<4x6xf32> {
    %0 = stablehlo.constant dense<[[-0.721201837, 2.23800898, 4.71029902, -3.00257468, 6.17815113, 2.06732202], [-3.94097209, 1.33419371, 0.69140327, -1.02265453, 2.20039272, 0.275565416], [4.30152416, -1.71299613, 0.263535082, -4.40533876, -3.60125089, -2.41911221], [-1.60160744, -0.360377938, -3.92941046, -1.72312105, 2.08186316, -2.28974199]]> : tensor<4x6xf32>
    return %0 : tensor<4x6xf32>
  }
  func.func private @expected() -> tensor<3x10xf32> {
    %0 = stablehlo.constant dense<[[-4.66217375, 3.57220268, 3.57220268, 5.4017024, 5.4017024, -4.02522945, -4.02522945, 8.37854385, 8.37854385, 2.3428874], [0.360552073, -0.378802419, -0.378802419, 0.954938352, 0.954938352, -5.4279933, -5.4279933, -1.40085816, -1.40085816, -2.14354682], [2.69991684, -2.07337403, -2.07337403, -3.66587543, -3.66587543, -6.128460e+00, -6.128460e+00, -1.51938772, -1.51938772, -4.7088542]]> : tensor<3x10xf32>
    return %0 : tensor<3x10xf32>
  }
}

