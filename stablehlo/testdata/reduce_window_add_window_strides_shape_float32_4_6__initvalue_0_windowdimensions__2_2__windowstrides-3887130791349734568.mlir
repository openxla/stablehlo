// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<4x6xf32>
    %1 = call @expected() : () -> tensor<3x3xf32>
    %2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f32>) -> tensor<f32>
    %4 = "stablehlo.reduce_window"(%0, %3) ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %6 = stablehlo.add %arg0, %arg1 : tensor<f32>
      stablehlo.return %6 : tensor<f32>
    }) {window_dimensions = dense<2> : tensor<2xi64>, window_strides = dense<[1, 2]> : tensor<2xi64>} : (tensor<4x6xf32>, tensor<f32>) -> tensor<3x3xf32>
    %5 = stablehlo.custom_call @check.eq(%4, %1) : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<i1>
    return %5 : tensor<i1>
  }
  func.func private @inputs() -> tensor<4x6xf32> {
    %0 = stablehlo.constant dense<[[5.04044056, -1.35050166, -0.989810109, -1.21777594, -3.44332719, -4.36912727], [0.324533284, -0.917935133, -2.68453097, 1.63836014, -0.478707731, 1.18392539], [-1.68799078, -0.0580824949, 3.42516971, -0.924332678, 2.1063211, -3.60422421], [-3.62437344, 4.18267059, -2.1504066, 4.36748314, -2.22356176, -2.91891503]]> : tensor<4x6xf32>
    return %0 : tensor<4x6xf32>
  }
  func.func private @expected() -> tensor<3x3xf32> {
    %0 = stablehlo.constant dense<[[3.09653735, -3.253757, -7.1072359], [-2.33947515, 1.45466614, -0.792685508], [-1.18777609, 4.71791363, -6.6403799]]> : tensor<3x3xf32>
    return %0 : tensor<3x3xf32>
  }
}

