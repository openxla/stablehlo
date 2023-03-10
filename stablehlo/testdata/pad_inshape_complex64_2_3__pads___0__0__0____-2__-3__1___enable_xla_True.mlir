// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<2x3xcomplex<f32>>, tensor<complex<f32>>)
    %1 = call @expected() : () -> tensor<2x0xcomplex<f32>>
    %2 = stablehlo.pad %0#0, %0#1, low = [0, -2], high = [0, -3], interior = [0, 1] : (tensor<2x3xcomplex<f32>>, tensor<complex<f32>>) -> tensor<2x0xcomplex<f32>>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<2x0xcomplex<f32>>, tensor<2x0xcomplex<f32>>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<2x3xcomplex<f32>>, tensor<complex<f32>>) {
    %0 = stablehlo.constant dense<[[(0.00151853461,-0.0019666017), (9.18364268E-4,3.44682718E-4), (8.57175735E-4,-0.00115322066)], [(-0.00111423363,5.54542698E-4), (0.00320347492,-1.84882156E-5), (-0.00193491625,-5.4825464E-4)]]> : tensor<2x3xcomplex<f32>>
    %1 = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<complex<f32>>
    return %0, %1 : tensor<2x3xcomplex<f32>>, tensor<complex<f32>>
  }
  func.func private @expected() -> tensor<2x0xcomplex<f32>> {
    %0 = stablehlo.constant dense<> : tensor<2x0xcomplex<f32>>
    return %0 : tensor<2x0xcomplex<f32>>
  }
}
