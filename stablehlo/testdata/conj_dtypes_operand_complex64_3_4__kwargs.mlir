// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<3x4xcomplex<f32>>
    %1 = call @expected() : () -> tensor<3x4xcomplex<f32>>
    %2 = stablehlo.real %0 : (tensor<3x4xcomplex<f32>>) -> tensor<3x4xf32>
    %3 = stablehlo.imag %0 : (tensor<3x4xcomplex<f32>>) -> tensor<3x4xf32>
    %4 = stablehlo.negate %3 : tensor<3x4xf32>
    %5 = stablehlo.complex %2, %4 : tensor<3x4xcomplex<f32>>
    %6 = stablehlo.custom_call @check.eq(%5, %1) : (tensor<3x4xcomplex<f32>>, tensor<3x4xcomplex<f32>>) -> tensor<i1>
    return %6 : tensor<i1>
  }
  func.func private @inputs() -> tensor<3x4xcomplex<f32>> {
    %0 = stablehlo.constant dense<[[(-0.956218242,1.94173658), (-4.1613307,-5.163280e+00), (3.80697155,-0.616726041), (4.39208317,4.55528641)], [(-1.34721899,0.898265779), (-0.006938498,-0.159980223), (-0.573709309,0.452298135), (3.15506649,-2.51832867)], [(4.23953533,2.43873072), (-4.782110e-01,4.59075451), (2.12633252,0.75286448), (4.727560e+00,0.232730106)]]> : tensor<3x4xcomplex<f32>>
    return %0 : tensor<3x4xcomplex<f32>>
  }
  func.func private @expected() -> tensor<3x4xcomplex<f32>> {
    %0 = stablehlo.constant dense<[[(-0.956218242,-1.94173658), (-4.1613307,5.163280e+00), (3.80697155,0.616726041), (4.39208317,-4.55528641)], [(-1.34721899,-0.898265779), (-0.006938498,0.159980223), (-0.573709309,-0.452298135), (3.15506649,2.51832867)], [(4.23953533,-2.43873072), (-4.782110e-01,-4.59075451), (2.12633252,-0.75286448), (4.727560e+00,-0.232730106)]]> : tensor<3x4xcomplex<f32>>
    return %0 : tensor<3x4xcomplex<f32>>
  }
}
