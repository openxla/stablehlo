// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<3x4x5xf32>
    %1 = call @expected() : () -> tensor<3x4x5xf32>
    %2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f32>) -> tensor<3x4x5xf32>
    %4 = stablehlo.custom_call @check.eq(%3, %1) : (tensor<3x4x5xf32>, tensor<3x4x5xf32>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> tensor<3x4x5xf32> {
    %0 = stablehlo.constant dense<[[[1.87026227, -0.811945796, 2.57706857, -0.289837748, 3.97441936], [-1.13554657, 0.415556967, -2.88057208, 1.18617833, -3.4486804], [-4.32041836, -2.94914722, 7.36368799, 1.38611364, 1.301368], [-0.767414927, -0.757800102, -0.464658827, 3.85030746, -2.59232831]], [[-0.530421615, 5.76317596, -2.97957563, -0.845373988, 6.51513147], [-1.02656734, -4.5334053, 3.38230491, 0.04679735, -2.92822671], [-3.80872035, 3.32863069, -0.578941286, 2.11730814, -7.48804903], [1.04939961, 1.11353207, -3.334203, 1.70561934, -2.90975356]], [[-0.400964439, -2.46582985, 6.16964244, 4.09752369, -1.21563268], [5.99286746, -3.45370173, -4.0987587, 0.281320572, -2.1342752], [4.947810e-01, 5.86681175, -2.26104784, -3.07913923, -0.191949829], [-3.420120e+00, 0.139410481, -1.28254235, 3.04297805, 6.0886774]]]> : tensor<3x4x5xf32>
    return %0 : tensor<3x4x5xf32>
  }
  func.func private @expected() -> tensor<3x4x5xf32> {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<3x4x5xf32>
    return %0 : tensor<3x4x5xf32>
  }
}
