// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<4x6xf32>, tensor<4x6xf32>)
    %1 = call @expected() : () -> tensor<4x14xf32>
    %2 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %3 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4:2 = "stablehlo.reduce_window"(%0#1, %0#0, %2, %3) ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<f32>, %arg3: tensor<f32>):
      %6 = stablehlo.compare  LE, %arg0, %arg2,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %7 = stablehlo.select %6, %arg0, %arg2 : tensor<i1>, tensor<f32>
      %8 = stablehlo.select %6, %arg1, %arg3 : tensor<i1>, tensor<f32>
      stablehlo.return %7, %8 : tensor<f32>, tensor<f32>
    }) {base_dilations = dense<[2, 3]> : tensor<2xi64>, window_dilations = dense<[3, 2]> : tensor<2xi64>, window_dimensions = dense<2> : tensor<2xi64>} : (tensor<4x6xf32>, tensor<4x6xf32>, tensor<f32>, tensor<f32>) -> (tensor<4x14xf32>, tensor<4x14xf32>)
    %5 = stablehlo.custom_call @check.eq(%4#1, %1) : (tensor<4x14xf32>, tensor<4x14xf32>) -> tensor<i1>
    return %5 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x6xf32>, tensor<4x6xf32>) {
    %0 = stablehlo.constant dense<[[-7.332500e+00, -2.8230865, -2.1608026, -6.48773718, 0.61284691, -3.96519828], [0.0465461574, 5.38721323, 1.25013793, -0.177122667, 1.92081726, 2.0936141], [0.236300573, -0.568100274, -3.77485967, 3.89434862, -0.0251923166, 2.02801013], [-3.10831499, 3.74822259, -1.96862555, 0.539672792, 0.686267554, 2.08048344]]> : tensor<4x6xf32>
    %1 = stablehlo.constant dense<[[3.10043883, 4.66068649, -2.81033468, -1.53667319, 1.77217698, -1.32437396], [-0.275190711, 1.21586978, -2.55467081, 2.4923737, 5.44136906, 2.098700e-01], [-5.34039307, 7.80149841, -0.282854199, 2.02033567, 4.23319483, 2.00467014], [1.78797388, 3.62742519, -1.86838973, 2.383600e+00, 1.63595808, -1.83778524]]> : tensor<4x6xf32>
    return %0, %1 : tensor<4x6xf32>, tensor<4x6xf32>
  }
  func.func private @expected() -> tensor<4x14xf32> {
    %0 = stablehlo.constant dense<[[-7.332500e+00, -2.8230865, 0.000000e+00, -2.8230865, -2.1608026, 0.000000e+00, -2.1608026, -6.48773718, 0.000000e+00, -6.48773718, 0.61284691, 0.000000e+00, 0.61284691, -3.96519828], [0.236300573, -0.568100274, 0.000000e+00, -0.568100274, -3.77485967, 0.000000e+00, -3.77485967, 3.89434862, 0.000000e+00, 3.89434862, -0.0251923166, 0.000000e+00, -0.0251923166, 2.02801013], [0.0465461574, 5.38721323, 0.000000e+00, 5.38721323, 1.25013793, 0.000000e+00, 1.25013793, -0.177122667, 0.000000e+00, -0.177122667, 1.92081726, 0.000000e+00, 1.92081726, 2.0936141], [-3.10831499, 3.74822259, 0.000000e+00, 3.74822259, -1.96862555, 0.000000e+00, -1.96862555, 0.539672792, 0.000000e+00, 0.539672792, 0.686267554, 0.000000e+00, 0.686267554, 2.08048344]]> : tensor<4x14xf32>
    return %0 : tensor<4x14xf32>
  }
}

