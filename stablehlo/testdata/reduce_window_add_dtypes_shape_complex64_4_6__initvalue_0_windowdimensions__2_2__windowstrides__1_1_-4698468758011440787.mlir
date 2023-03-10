// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<4x6xcomplex<f32>>
    %1 = call @expected() : () -> tensor<3x5xcomplex<f32>>
    %2 = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<complex<f32>>
    %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<complex<f32>>) -> tensor<complex<f32>>
    %4 = "stablehlo.reduce_window"(%0, %3) ({
    ^bb0(%arg0: tensor<complex<f32>>, %arg1: tensor<complex<f32>>):
      %6 = stablehlo.add %arg0, %arg1 : tensor<complex<f32>>
      stablehlo.return %6 : tensor<complex<f32>>
    }) {window_dimensions = dense<2> : tensor<2xi64>} : (tensor<4x6xcomplex<f32>>, tensor<complex<f32>>) -> tensor<3x5xcomplex<f32>>
    %5 = stablehlo.custom_call @check.eq(%4, %1) : (tensor<3x5xcomplex<f32>>, tensor<3x5xcomplex<f32>>) -> tensor<i1>
    return %5 : tensor<i1>
  }
  func.func private @inputs() -> tensor<4x6xcomplex<f32>> {
    %0 = stablehlo.constant dense<[[(-1.1183852,-8.75956535), (4.95623255,1.03840244), (-2.6998961,-1.47844553), (1.52361262,-2.13585711), (-2.71564484,-3.01033831), (-2.35609889,-1.31232381)], [(-4.3393445,1.53966928), (2.45947385,4.88709259), (1.71561635,1.78098798), (-1.58346224,0.185394287), (-2.03025436,-1.38893974), (0.279954791,-4.70554352)], [(1.31765366,-1.31122053), (-5.65778637,-0.334982365), (-0.325271189,1.39524794), (-4.62310791,-0.242266238), (-0.450628936,1.06363821), (-0.16774942,4.57756615)], [(1.89521587,-0.220859662), (-3.95998883,0.765856385), (1.97782564,-2.14821935), (-4.48547316,-0.0708763599), (-3.93513608,-0.684421301), (0.588392735,1.47578061)]]> : tensor<4x6xcomplex<f32>>
    return %0 : tensor<4x6xcomplex<f32>>
  }
  func.func private @expected() -> tensor<3x5xcomplex<f32>> {
    %0 = stablehlo.constant dense<[[(1.95797658,-1.29440117), (6.43142653,6.22803783), (-1.04412937,-1.64792037), (-4.80574894,-6.34974098), (-6.82204341,-10.4171457)], [(-6.22000313,4.78055859), (-1.80796719,7.72834587), (-4.81622505,3.11936402), (-8.68745326,-0.382173538), (-2.36867785,-0.453279018)], [(-6.40490532,-1.10120618), (-7.96522093,-0.322097421), (-7.45602655,-1.06611395), (-13.4943457,0.0660743117), (-3.96512175,6.43256378)]]> : tensor<3x5xcomplex<f32>>
    return %0 : tensor<3x5xcomplex<f32>>
  }
}

