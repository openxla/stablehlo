// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<3x4xf32>
    %1 = call @expected() : () -> tensor<3x4xcomplex<f32>>
    %2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f32>) -> tensor<3x4xf32>
    %4 = stablehlo.complex %0, %3 : tensor<3x4xcomplex<f32>>
    %5 = stablehlo.custom_call @check.eq(%4, %1) : (tensor<3x4xcomplex<f32>>, tensor<3x4xcomplex<f32>>) -> tensor<i1>
    return %5 : tensor<i1>
  }
  func.func private @inputs() -> tensor<3x4xf32> {
    %0 = stablehlo.constant dense<[[-0.731090128, -0.437817723, -8.06553649, 6.15966654], [7.79678106, 4.86321259, -2.22347236, -2.76911235], [-1.0445224, 1.34840381, -4.22374058, -1.58598387]]> : tensor<3x4xf32>
    return %0 : tensor<3x4xf32>
  }
  func.func private @expected() -> tensor<3x4xcomplex<f32>> {
    %0 = stablehlo.constant dense<[[(-0.731090128,0.000000e+00), (-0.437817723,0.000000e+00), (-8.06553649,0.000000e+00), (6.15966654,0.000000e+00)], [(7.79678106,0.000000e+00), (4.86321259,0.000000e+00), (-2.22347236,0.000000e+00), (-2.76911235,0.000000e+00)], [(-1.0445224,0.000000e+00), (1.34840381,0.000000e+00), (-4.22374058,0.000000e+00), (-1.58598387,0.000000e+00)]]> : tensor<3x4xcomplex<f32>>
    return %0 : tensor<3x4xcomplex<f32>>
  }
}
