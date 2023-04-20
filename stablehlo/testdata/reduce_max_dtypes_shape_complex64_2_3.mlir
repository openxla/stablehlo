// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<2x3xcomplex<f32>>
    %1 = call @expected() : () -> tensor<3xcomplex<f32>>
    %2 = stablehlo.constant dense<(0xFF800000,0.000000e+00)> : tensor<complex<f32>>
    %3 = stablehlo.reduce(%0 init: %2) across dimensions = [0] : (tensor<2x3xcomplex<f32>>, tensor<complex<f32>>) -> tensor<3xcomplex<f32>>
     reducer(%arg0: tensor<complex<f32>>, %arg1: tensor<complex<f32>>)  {
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
    }
    %4 = stablehlo.custom_call @check.eq(%3, %1) : (tensor<3xcomplex<f32>>, tensor<3xcomplex<f32>>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> tensor<2x3xcomplex<f32>> {
    %0 = stablehlo.constant dense<[[(-0.766099572,-1.75491714), (-0.704843879,-0.456715196), (-0.449349642,-4.39625645)], [(-0.467202455,5.20890427), (-4.38879538,0.622361779), (4.78189754,-0.113236174)]]> : tensor<2x3xcomplex<f32>>
    return %0 : tensor<2x3xcomplex<f32>>
  }
  func.func private @expected() -> tensor<3xcomplex<f32>> {
    %0 = stablehlo.constant dense<[(-0.467202455,5.20890427), (-0.704843879,-0.456715196), (4.78189754,-0.113236174)]> : tensor<3xcomplex<f32>>
    return %0 : tensor<3xcomplex<f32>>
  }
}
