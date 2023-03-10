// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<4x6xcomplex<f32>>
    %1 = call @expected() : () -> tensor<3x5xcomplex<f32>>
    %2 = stablehlo.constant dense<(1.000000e+00,0.000000e+00)> : tensor<complex<f32>>
    %3 = "stablehlo.reduce_window"(%0, %2) ({
    ^bb0(%arg0: tensor<complex<f32>>, %arg1: tensor<complex<f32>>):
      %5 = stablehlo.real %arg0 : (tensor<complex<f32>>) -> tensor<f32>
      %6 = stablehlo.real %arg1 : (tensor<complex<f32>>) -> tensor<f32>
      %7 = stablehlo.compare  EQ, %5, %6,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %8 = stablehlo.compare  GT, %5, %6,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %9 = stablehlo.imag %arg0 : (tensor<complex<f32>>) -> tensor<f32>
      %10 = stablehlo.imag %arg1 : (tensor<complex<f32>>) -> tensor<f32>
      %11 = stablehlo.compare  GT, %9, %10,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %12 = stablehlo.select %7, %11, %8 : tensor<i1>, tensor<i1>
      %13 = stablehlo.select %12, %arg0, %arg1 : tensor<i1>, tensor<complex<f32>>
      stablehlo.return %13 : tensor<complex<f32>>
    }) {window_dimensions = dense<2> : tensor<2xi64>} : (tensor<4x6xcomplex<f32>>, tensor<complex<f32>>) -> tensor<3x5xcomplex<f32>>
    %4 = stablehlo.custom_call @check.eq(%3, %1) : (tensor<3x5xcomplex<f32>>, tensor<3x5xcomplex<f32>>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> tensor<4x6xcomplex<f32>> {
    %0 = stablehlo.constant dense<[[(-0.972649574,0.672472298), (4.43000078,-0.0616510063), (-0.676460206,-3.93928528), (0.299077809,2.24311066), (3.42091846,3.299890e-01), (2.11727023,1.71635234)], [(-3.62902927,4.08627224), (-4.85351563,-1.09559417), (-0.360351264,-6.29636049), (2.40622211,-3.73725653), (-2.92994714,-2.01690793), (-3.29381609,-2.40824866)], [(2.0407269,-1.92697251), (-2.2930243,0.825138807), (-2.60777473,-2.42724395), (4.80422306,2.66609406), (-0.844612717,-0.964456975), (-0.460476875,-1.79708683)], [(3.09630489,2.76739264), (2.39456558,-1.19222534), (0.837192297,-0.0854947641), (-0.661449134,3.7603271), (6.18411779,6.11016512), (0.861008107,0.411064506)]]> : tensor<4x6xcomplex<f32>>
    return %0 : tensor<4x6xcomplex<f32>>
  }
  func.func private @expected() -> tensor<3x5xcomplex<f32>> {
    %0 = stablehlo.constant dense<[[(4.43000078,-0.0616510063), (4.43000078,-0.0616510063), (2.40622211,-3.73725653), (3.42091846,3.299890e-01), (3.42091846,3.299890e-01)], [(2.0407269,-1.92697251), (1.000000e+00,0.000000e+00), (4.80422306,2.66609406), (4.80422306,2.66609406), (1.000000e+00,0.000000e+00)], [(3.09630489,2.76739264), (2.39456558,-1.19222534), (4.80422306,2.66609406), (6.18411779,6.11016512), (6.18411779,6.11016512)]]> : tensor<3x5xcomplex<f32>>
    return %0 : tensor<3x5xcomplex<f32>>
  }
}

