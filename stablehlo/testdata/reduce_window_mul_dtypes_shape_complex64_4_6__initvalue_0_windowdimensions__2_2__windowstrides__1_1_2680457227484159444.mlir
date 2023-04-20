// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<4x6xcomplex<f32>>
    %1 = call @expected() : () -> tensor<3x5xcomplex<f32>>
    %2 = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<complex<f32>>
    %3 = "stablehlo.reduce_window"(%0, %2) ({
    ^bb0(%arg0: tensor<complex<f32>>, %arg1: tensor<complex<f32>>):
      %5 = stablehlo.multiply %arg0, %arg1 : tensor<complex<f32>>
      stablehlo.return %5 : tensor<complex<f32>>
    }) {window_dimensions = dense<2> : tensor<2xi64>} : (tensor<4x6xcomplex<f32>>, tensor<complex<f32>>) -> tensor<3x5xcomplex<f32>>
    %4 = stablehlo.custom_call @check.eq(%3, %1) : (tensor<3x5xcomplex<f32>>, tensor<3x5xcomplex<f32>>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> tensor<4x6xcomplex<f32>> {
    %0 = stablehlo.constant dense<[[(3.92327094,2.69720411), (-1.34484529,-3.37101912), (3.09930229,-2.93890905), (1.12214816,-0.242806554), (0.720924258,-0.326681048), (-0.818712055,-1.46094155)], [(5.29841375,1.64236474), (0.215576604,3.64783621), (5.381390e+00,-4.77757883), (-0.896658718,2.79174042), (-0.260070264,-1.3840338), (-2.25516677,-4.30485535)], [(1.24443519,1.94716597), (-2.87089825,2.79661202), (1.51471436,1.09331846), (6.09001493,2.41639161), (-5.4137187,-2.76647854), (3.69677138,4.28614807)], [(-1.63428807,-0.261969656), (-1.27768898,-2.56426978), (1.68950355,-2.7240808), (-3.46054363,-2.42063618), (-1.26553774,0.469694018), (-0.387948632,4.58789492)]]> : tensor<4x6xcomplex<f32>>
    return %0 : tensor<4x6xcomplex<f32>>
  }
  func.func private @expected() -> tensor<3x5xcomplex<f32>> {
    %0 = stablehlo.constant dense<[[(0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00), (-0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00)], [(-0.000000e+00,0.000000e+00), (-0.000000e+00,0.000000e+00), (-0.000000e+00,0.000000e+00), (0.000000e+00,-0.000000e+00), (0.000000e+00,0.000000e+00)], [(0.000000e+00,-0.000000e+00), (0.000000e+00,0.000000e+00), (0.000000e+00,-0.000000e+00), (0.000000e+00,-0.000000e+00), (0.000000e+00,-0.000000e+00)]]> : tensor<3x5xcomplex<f32>>
    return %0 : tensor<3x5xcomplex<f32>>
  }
}

