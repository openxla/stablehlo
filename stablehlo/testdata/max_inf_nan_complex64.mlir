// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<3x3xcomplex<f32>>, tensor<3x3xcomplex<f32>>)
    %1 = call @expected() : () -> tensor<3x3xcomplex<f32>>
    %2 = stablehlo.real %0#0 : (tensor<3x3xcomplex<f32>>) -> tensor<3x3xf32>
    %3 = stablehlo.real %0#1 : (tensor<3x3xcomplex<f32>>) -> tensor<3x3xf32>
    %4 = stablehlo.compare  EQ, %2, %3,  FLOAT : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x3xi1>
    %5 = stablehlo.compare  GT, %2, %3,  FLOAT : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x3xi1>
    %6 = stablehlo.imag %0#0 : (tensor<3x3xcomplex<f32>>) -> tensor<3x3xf32>
    %7 = stablehlo.imag %0#1 : (tensor<3x3xcomplex<f32>>) -> tensor<3x3xf32>
    %8 = stablehlo.compare  GT, %6, %7,  FLOAT : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x3xi1>
    %9 = stablehlo.select %4, %8, %5 : tensor<3x3xi1>, tensor<3x3xi1>
    %10 = stablehlo.select %9, %0#0, %0#1 : tensor<3x3xi1>, tensor<3x3xcomplex<f32>>
    %11 = stablehlo.custom_call @check.eq(%10, %1) : (tensor<3x3xcomplex<f32>>, tensor<3x3xcomplex<f32>>) -> tensor<i1>
    return %11 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<3x3xcomplex<f32>>, tensor<3x3xcomplex<f32>>) {
    %0 = stablehlo.constant dense<[[(0x7FC00000,0.000000e+00), (0x7FC00000,0.000000e+00), (0x7FC00000,0.000000e+00)], [(0x7F800000,0.000000e+00), (0x7F800000,0.000000e+00), (0x7F800000,0.000000e+00)], [(0xFF800000,0.000000e+00), (0xFF800000,0.000000e+00), (0xFF800000,0.000000e+00)]]> : tensor<3x3xcomplex<f32>>
    %1 = stablehlo.constant dense<[[(0x7FC00000,0.000000e+00), (0x7F800000,0.000000e+00), (0xFF800000,0.000000e+00)], [(0x7FC00000,0.000000e+00), (0x7F800000,0.000000e+00), (0xFF800000,0.000000e+00)], [(0x7FC00000,0.000000e+00), (0x7F800000,0.000000e+00), (0xFF800000,0.000000e+00)]]> : tensor<3x3xcomplex<f32>>
    return %0, %1 : tensor<3x3xcomplex<f32>>, tensor<3x3xcomplex<f32>>
  }
  func.func private @expected() -> tensor<3x3xcomplex<f32>> {
    %0 = stablehlo.constant dense<[[(0x7FC00000,0.000000e+00), (0x7F800000,0.000000e+00), (0xFF800000,0.000000e+00)], [(0x7FC00000,0.000000e+00), (0x7F800000,0.000000e+00), (0x7F800000,0.000000e+00)], [(0x7FC00000,0.000000e+00), (0x7F800000,0.000000e+00), (0xFF800000,0.000000e+00)]]> : tensor<3x3xcomplex<f32>>
    return %0 : tensor<3x3xcomplex<f32>>
  }
}
