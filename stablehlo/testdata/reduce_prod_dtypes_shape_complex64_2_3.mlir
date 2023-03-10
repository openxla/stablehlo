// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<2x3xcomplex<f32>>
    %1 = call @expected() : () -> tensor<3xcomplex<f32>>
    %2 = stablehlo.constant dense<(1.000000e+00,0.000000e+00)> : tensor<complex<f32>>
    %3 = stablehlo.reduce(%0 init: %2) across dimensions = [0] : (tensor<2x3xcomplex<f32>>, tensor<complex<f32>>) -> tensor<3xcomplex<f32>>
     reducer(%arg0: tensor<complex<f32>>, %arg1: tensor<complex<f32>>)  {
      %5 = stablehlo.multiply %arg0, %arg1 : tensor<complex<f32>>
      stablehlo.return %5 : tensor<complex<f32>>
    }
    %4 = stablehlo.custom_call @check.eq(%3, %1) : (tensor<3xcomplex<f32>>, tensor<3xcomplex<f32>>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> tensor<2x3xcomplex<f32>> {
    %0 = stablehlo.constant dense<[[(-4.78967524,-0.294267416), (-1.80710745,2.25319576), (-4.19134426,-3.72368598)], [(1.81930077,-4.255660e+00), (-1.1868422,1.64352047), (2.66921759,0.870566308)]]> : tensor<2x3xcomplex<f32>>
    return %0 : tensor<2x3xcomplex<f32>>
  }
  func.func private @expected() -> tensor<3xcomplex<f32>> {
    %0 = stablehlo.constant dense<[(-9.96616172,19.8478699), (-1.55842209,-5.64420605), (-7.94589424,-13.588171)]> : tensor<3xcomplex<f32>>
    return %0 : tensor<3xcomplex<f32>>
  }
}
