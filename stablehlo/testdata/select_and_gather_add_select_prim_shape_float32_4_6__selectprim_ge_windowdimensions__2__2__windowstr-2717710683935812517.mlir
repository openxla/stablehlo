// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<4x6xf32>, tensor<4x6xf32>)
    %1 = call @expected() : () -> tensor<3x5xf32>
    %2 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %3 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %4:2 = "stablehlo.reduce_window"(%0#1, %0#0, %2, %3) ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<f32>, %arg3: tensor<f32>):
      %6 = stablehlo.compare  GE, %arg0, %arg2,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %7 = stablehlo.select %6, %arg0, %arg2 : tensor<i1>, tensor<f32>
      %8 = stablehlo.select %6, %arg1, %arg3 : tensor<i1>, tensor<f32>
      stablehlo.return %7, %8 : tensor<f32>, tensor<f32>
    }) {window_dimensions = dense<2> : tensor<2xi64>} : (tensor<4x6xf32>, tensor<4x6xf32>, tensor<f32>, tensor<f32>) -> (tensor<3x5xf32>, tensor<3x5xf32>)
    %5 = stablehlo.custom_call @check.eq(%4#1, %1) : (tensor<3x5xf32>, tensor<3x5xf32>) -> tensor<i1>
    return %5 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x6xf32>, tensor<4x6xf32>) {
    %0 = stablehlo.constant dense<[[5.57956886, -4.13894558, 1.145805, -1.9771781, 3.54417777, -3.09877872], [3.30714726, 2.48644686, -1.43284643, -1.09521699, -7.34765816, -5.94207907], [2.56617498, -2.68347812, 1.35620379, -4.32303715, -0.0655869469, 3.57533669], [-2.05537534, -1.54444289, 3.38338327, 4.09955406, -2.80702162, 1.7128011]]> : tensor<4x6xf32>
    %1 = stablehlo.constant dense<[[2.16599584, 0.431703299, 1.08035028, -1.76984024, -8.115790e-01, 1.85179114], [6.33482313, -9.631990e-01, 0.947742402, -2.23918796, 7.207060e-02, -1.79294395], [3.40710807, -5.99408293, 3.80985904, 2.15884924, -2.954480e+00, 4.1574707], [-1.44822037, 6.13582897, 2.83970404, 1.46105516, 1.31010675, -1.70051467]]> : tensor<4x6xf32>
    return %0, %1 : tensor<4x6xf32>, tensor<4x6xf32>
  }
  func.func private @expected() -> tensor<3x5xf32> {
    %0 = stablehlo.constant dense<[[3.30714726, 1.145805, 1.145805, -7.34765816, -3.09877872], [3.30714726, 1.35620379, 1.35620379, -4.32303715, 3.57533669], [-1.54444289, -1.54444289, 1.35620379, -4.32303715, 3.57533669]]> : tensor<3x5xf32>
    return %0 : tensor<3x5xf32>
  }
}

