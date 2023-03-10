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
    %0 = stablehlo.constant dense<[[-4.51149464, -0.455049247, -1.78570271, 0.99308896], [1.92249668, 0.0967779085, 3.67370224, 0.61528176], [-0.424995452, -2.40207338, 1.82864797, 0.292340934]]> : tensor<3x4xf32>
    return %0 : tensor<3x4xf32>
  }
  func.func private @expected() -> tensor<3x4xcomplex<f32>> {
    %0 = stablehlo.constant dense<[[(-4.51149464,0.000000e+00), (-0.455049247,0.000000e+00), (-1.78570271,0.000000e+00), (0.99308896,0.000000e+00)], [(1.92249668,0.000000e+00), (0.0967779085,0.000000e+00), (3.67370224,0.000000e+00), (0.61528176,0.000000e+00)], [(-0.424995452,0.000000e+00), (-2.40207338,0.000000e+00), (1.82864797,0.000000e+00), (0.292340934,0.000000e+00)]]> : tensor<3x4xcomplex<f32>>
    return %0 : tensor<3x4xcomplex<f32>>
  }
}
