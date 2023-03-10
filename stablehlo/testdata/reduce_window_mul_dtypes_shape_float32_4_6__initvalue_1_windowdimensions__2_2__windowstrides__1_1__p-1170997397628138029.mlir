// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<4x6xf32>
    %1 = call @expected() : () -> tensor<3x5xf32>
    %2 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %3 = "stablehlo.reduce_window"(%0, %2) ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %5 = stablehlo.multiply %arg0, %arg1 : tensor<f32>
      stablehlo.return %5 : tensor<f32>
    }) {window_dimensions = dense<2> : tensor<2xi64>} : (tensor<4x6xf32>, tensor<f32>) -> tensor<3x5xf32>
    %4 = stablehlo.custom_call @check.eq(%3, %1) : (tensor<3x5xf32>, tensor<3x5xf32>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> tensor<4x6xf32> {
    %0 = stablehlo.constant dense<[[0.835746169, 7.91090918, -2.29111934, -6.8987813, -2.69525576, -4.32770443], [-1.31710482, 5.32159424, 1.23298335, -1.21830738, 1.76821625, -3.3130722], [-7.36848402, 0.885791957, -0.956944406, -3.20059419, 1.21716928, 3.25353551], [0.958624184, -3.26040292, -1.753500e-01, -1.75717437, -3.61557221, -3.00892448]]> : tensor<4x6xf32>
    return %0 : tensor<4x6xf32>
  }
  func.func private @expected() -> tensor<3x5xf32> {
    %0 = stablehlo.constant dense<[[-46.3407326, -118.924988, -23.7429218, -40.0557289, -68.3319549], [45.7479858, -5.56182623, -4.60077953, 8.39216136, -23.1991901], [20.3999691, -0.484613478, 0.943708658, -24.7499027, 4.308190e+01]]> : tensor<3x5xf32>
    return %0 : tensor<3x5xf32>
  }
}

