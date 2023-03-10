// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<2x3xcomplex<f32>>, tensor<complex<f32>>)
    %1 = call @expected() : () -> tensor<2x3xcomplex<f32>>
    %2 = stablehlo.pad %0#0, %0#1, low = [0, 0], high = [0, 0], interior = [0, 0] : (tensor<2x3xcomplex<f32>>, tensor<complex<f32>>) -> tensor<2x3xcomplex<f32>>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<2x3xcomplex<f32>>, tensor<2x3xcomplex<f32>>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<2x3xcomplex<f32>>, tensor<complex<f32>>) {
    %0 = stablehlo.constant dense<[[(-8.24017406E-5,-0.00155910291), (-0.00220226985,-2.4060272E-4), (0.00150859717,-4.31594672E-4)], [(-3.6077312E-4,9.13478376E-4), (-4.26453567E-4,6.341860e-04), (1.94213044E-6,5.68056654E-4)]]> : tensor<2x3xcomplex<f32>>
    %1 = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<complex<f32>>
    return %0, %1 : tensor<2x3xcomplex<f32>>, tensor<complex<f32>>
  }
  func.func private @expected() -> tensor<2x3xcomplex<f32>> {
    %0 = stablehlo.constant dense<[[(-8.24017406E-5,-0.00155910291), (-0.00220226985,-2.4060272E-4), (0.00150859717,-4.31594672E-4)], [(-3.6077312E-4,9.13478376E-4), (-4.26453567E-4,6.341860e-04), (1.94213044E-6,5.68056654E-4)]]> : tensor<2x3xcomplex<f32>>
    return %0 : tensor<2x3xcomplex<f32>>
  }
}
