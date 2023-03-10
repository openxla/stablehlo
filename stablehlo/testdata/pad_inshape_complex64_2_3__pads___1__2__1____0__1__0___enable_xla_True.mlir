// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<2x3xcomplex<f32>>, tensor<complex<f32>>)
    %1 = call @expected() : () -> tensor<6x4xcomplex<f32>>
    %2 = stablehlo.pad %0#0, %0#1, low = [1, 0], high = [2, 1], interior = [1, 0] : (tensor<2x3xcomplex<f32>>, tensor<complex<f32>>) -> tensor<6x4xcomplex<f32>>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<6x4xcomplex<f32>>, tensor<6x4xcomplex<f32>>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<2x3xcomplex<f32>>, tensor<complex<f32>>) {
    %0 = stablehlo.constant dense<[[(2.73689628E-4,-0.00191447895), (6.28948153E-4,-1.05097541E-4), (3.03907203E-4,-0.00122220814)], [(4.28868316E-6,-0.00125718431), (-6.42397907E-4,-2.272930e-04), (-1.5718113E-4,-0.00136294332)]]> : tensor<2x3xcomplex<f32>>
    %1 = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<complex<f32>>
    return %0, %1 : tensor<2x3xcomplex<f32>>, tensor<complex<f32>>
  }
  func.func private @expected() -> tensor<6x4xcomplex<f32>> {
    %0 = stablehlo.constant dense<[[(0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00)], [(2.73689628E-4,-0.00191447895), (6.28948153E-4,-1.05097541E-4), (3.03907203E-4,-0.00122220814), (0.000000e+00,0.000000e+00)], [(0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00)], [(4.28868316E-6,-0.00125718431), (-6.42397907E-4,-2.272930e-04), (-1.5718113E-4,-0.00136294332), (0.000000e+00,0.000000e+00)], [(0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00)], [(0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00)]]> : tensor<6x4xcomplex<f32>>
    return %0 : tensor<6x4xcomplex<f32>>
  }
}
