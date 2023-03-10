// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<4x6xf32>, tensor<4x6xf32>)
    %1 = call @expected() : () -> tensor<4x6xf32>
    %2 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %3 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4:2 = "stablehlo.reduce_window"(%0#1, %0#0, %2, %3) ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<f32>, %arg3: tensor<f32>):
      %6 = stablehlo.compare  LE, %arg0, %arg2,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %7 = stablehlo.select %6, %arg0, %arg2 : tensor<i1>, tensor<f32>
      %8 = stablehlo.select %6, %arg1, %arg3 : tensor<i1>, tensor<f32>
      stablehlo.return %7, %8 : tensor<f32>, tensor<f32>
    }) {padding = dense<[[0, 1], [0, 1]]> : tensor<2x2xi64>, window_dimensions = dense<2> : tensor<2xi64>} : (tensor<4x6xf32>, tensor<4x6xf32>, tensor<f32>, tensor<f32>) -> (tensor<4x6xf32>, tensor<4x6xf32>)
    %5 = stablehlo.custom_call @check.eq(%4#1, %1) : (tensor<4x6xf32>, tensor<4x6xf32>) -> tensor<i1>
    return %5 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x6xf32>, tensor<4x6xf32>) {
    %0 = stablehlo.constant dense<[[3.14681983, -5.51675844, 2.99068975, -0.204112172, -1.44421732, 3.47418976], [-6.21619177, 4.93372822, 2.12878799, 0.606317341, 2.56640744, 2.657340e+00], [-3.34142399, 0.832507729, -1.75170922, -2.78363848, 0.191280186, -4.62775707], [-1.25499809, -4.23446321, -1.39376664, -0.814940214, -2.1730969, -3.380934]]> : tensor<4x6xf32>
    %1 = stablehlo.constant dense<[[-0.812187135, -4.81235552, 1.05026627, 0.511906743, 0.731464564, -0.558087289], [-0.645616471, 1.11495852, 0.400145292, -1.55490339, 1.95693386, 2.95456076], [2.99821734, -4.96495485, 0.929495632, 2.83182025, -0.0167944226, 3.779181], [4.07046604, -3.19877601, -2.82647538, -1.68686426, -4.05062437, -10.7977018]]> : tensor<4x6xf32>
    return %0, %1 : tensor<4x6xf32>, tensor<4x6xf32>
  }
  func.func private @expected() -> tensor<4x6xf32> {
    %0 = stablehlo.constant dense<[[-5.51675844, -5.51675844, 0.606317341, 0.606317341, 3.47418976, 3.47418976], [0.832507729, 0.832507729, 0.606317341, 0.606317341, 0.191280186, 2.657340e+00], [0.832507729, 0.832507729, -1.39376664, -2.1730969, -3.380934, -3.380934], [-4.23446321, -4.23446321, -1.39376664, -2.1730969, -3.380934, -3.380934]]> : tensor<4x6xf32>
    return %0 : tensor<4x6xf32>
  }
}

