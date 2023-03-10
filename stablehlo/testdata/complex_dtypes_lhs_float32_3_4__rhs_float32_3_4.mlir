// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<3x4xf32>, tensor<3x4xf32>)
    %1 = call @expected() : () -> tensor<3x4xcomplex<f32>>
    %2 = stablehlo.complex %0#0, %0#1 : tensor<3x4xcomplex<f32>>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<3x4xcomplex<f32>>, tensor<3x4xcomplex<f32>>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<3x4xf32>, tensor<3x4xf32>) {
    %0 = stablehlo.constant dense<[[-5.07035875, -3.72310281, -4.24285221, 4.083100e+00], [-3.61923885, -2.82356691, 7.91711425, -0.592124462], [-5.61838675, 5.38161421, -2.99476314, 1.32623148]]> : tensor<3x4xf32>
    %1 = stablehlo.constant dense<[[-1.51469374, 0.660175204, -2.72695398, -1.17417383], [3.38128543, 0.810583353, -1.625512, 2.66601849], [4.05134487, 4.48275757, -2.34912467, -0.595442355]]> : tensor<3x4xf32>
    return %0, %1 : tensor<3x4xf32>, tensor<3x4xf32>
  }
  func.func private @expected() -> tensor<3x4xcomplex<f32>> {
    %0 = stablehlo.constant dense<[[(-5.07035875,-1.51469374), (-3.72310281,0.660175204), (-4.24285221,-2.72695398), (4.083100e+00,-1.17417383)], [(-3.61923885,3.38128543), (-2.82356691,0.810583353), (7.91711425,-1.625512), (-0.592124462,2.66601849)], [(-5.61838675,4.05134487), (5.38161421,4.48275757), (-2.99476314,-2.34912467), (1.32623148,-0.595442355)]]> : tensor<3x4xcomplex<f32>>
    return %0 : tensor<3x4xcomplex<f32>>
  }
}
