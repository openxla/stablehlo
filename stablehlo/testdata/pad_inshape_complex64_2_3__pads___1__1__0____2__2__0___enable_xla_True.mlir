// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<2x3xcomplex<f32>>, tensor<complex<f32>>)
    %1 = call @expected() : () -> tensor<4x7xcomplex<f32>>
    %2 = stablehlo.pad %0#0, %0#1, low = [1, 2], high = [1, 2], interior = [0, 0] : (tensor<2x3xcomplex<f32>>, tensor<complex<f32>>) -> tensor<4x7xcomplex<f32>>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<4x7xcomplex<f32>>, tensor<4x7xcomplex<f32>>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<2x3xcomplex<f32>>, tensor<complex<f32>>) {
    %0 = stablehlo.constant dense<[[(-5.90622483E-4,9.75612085E-4), (-0.00117100554,5.6535576E-4), (-0.00142354018,-0.00116211327)], [(-3.46248795E-4,0.00154530944), (2.39170418E-4,4.51020926E-4), (-0.00203824905,2.21181108E-4)]]> : tensor<2x3xcomplex<f32>>
    %1 = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<complex<f32>>
    return %0, %1 : tensor<2x3xcomplex<f32>>, tensor<complex<f32>>
  }
  func.func private @expected() -> tensor<4x7xcomplex<f32>> {
    %0 = stablehlo.constant dense<[[(0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00)], [(0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00), (-5.90622483E-4,9.75612085E-4), (-0.00117100554,5.6535576E-4), (-0.00142354018,-0.00116211327), (0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00)], [(0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00), (-3.46248795E-4,0.00154530944), (2.39170418E-4,4.51020926E-4), (-0.00203824905,2.21181108E-4), (0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00)], [(0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00)]]> : tensor<4x7xcomplex<f32>>
    return %0 : tensor<4x7xcomplex<f32>>
  }
}
