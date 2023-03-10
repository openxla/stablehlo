// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<2x3xcomplex<f32>>, tensor<complex<f32>>)
    %1 = call @expected() : () -> tensor<2x1xcomplex<f32>>
    %2 = stablehlo.pad %0#0, %0#1, low = [0, -1], high = [0, -1], interior = [0, 0] : (tensor<2x3xcomplex<f32>>, tensor<complex<f32>>) -> tensor<2x1xcomplex<f32>>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<2x1xcomplex<f32>>, tensor<2x1xcomplex<f32>>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<2x3xcomplex<f32>>, tensor<complex<f32>>) {
    %0 = stablehlo.constant dense<[[(0.00159131433,0.00277683837), (0.00127878529,0.00111579127), (-7.43121854E-5,4.5982108E-4)], [(-0.00104789285,-3.30286712E-4), (-9.8787932E-5,6.21758576E-4), (0.00122523599,-5.91346819E-4)]]> : tensor<2x3xcomplex<f32>>
    %1 = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<complex<f32>>
    return %0, %1 : tensor<2x3xcomplex<f32>>, tensor<complex<f32>>
  }
  func.func private @expected() -> tensor<2x1xcomplex<f32>> {
    %0 = stablehlo.constant dense<[[(0.00127878529,0.00111579127)], [(-9.8787932E-5,6.21758576E-4)]]> : tensor<2x1xcomplex<f32>>
    return %0 : tensor<2x1xcomplex<f32>>
  }
}
