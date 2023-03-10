// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<2x3xcomplex<f32>>, tensor<complex<f32>>)
    %1 = call @expected() : () -> tensor<2x7xcomplex<f32>>
    %2 = stablehlo.pad %0#0, %0#1, low = [0, -2], high = [0, -2], interior = [0, 4] : (tensor<2x3xcomplex<f32>>, tensor<complex<f32>>) -> tensor<2x7xcomplex<f32>>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<2x7xcomplex<f32>>, tensor<2x7xcomplex<f32>>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<2x3xcomplex<f32>>, tensor<complex<f32>>) {
    %0 = stablehlo.constant dense<[[(5.31037396E-4,-0.00115685537), (-3.53360112E-4,-0.00138103205), (-4.89086378E-4,0.00177958969)], [(6.86371582E-4,2.33959916E-4), (-0.00113211747,-0.00123034476), (-0.00189835706,0.00120960351)]]> : tensor<2x3xcomplex<f32>>
    %1 = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<complex<f32>>
    return %0, %1 : tensor<2x3xcomplex<f32>>, tensor<complex<f32>>
  }
  func.func private @expected() -> tensor<2x7xcomplex<f32>> {
    %0 = stablehlo.constant dense<[[(0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00), (-3.53360112E-4,-0.00138103205), (0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00)], [(0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00), (-0.00113211747,-0.00123034476), (0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00)]]> : tensor<2x7xcomplex<f32>>
    return %0 : tensor<2x7xcomplex<f32>>
  }
}
