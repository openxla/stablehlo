// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<3x4xcomplex<f32>>, tensor<4x2xcomplex<f32>>)
    %1 = call @expected() : () -> tensor<3x2xcomplex<f32>>
    %2 = "stablehlo.dot_general"(%0#0, %0#1) {dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>, precision_config = [#stablehlo<precision HIGHEST>, #stablehlo<precision HIGHEST>]} : (tensor<3x4xcomplex<f32>>, tensor<4x2xcomplex<f32>>) -> tensor<3x2xcomplex<f32>>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<3x2xcomplex<f32>>, tensor<3x2xcomplex<f32>>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<3x4xcomplex<f32>>, tensor<4x2xcomplex<f32>>) {
    %0 = stablehlo.constant dense<[[(3.21495652,0.590251744), (-4.69663858,-1.13666058), (0.192925826,-2.84668732), (1.97103286,3.68735051)], [(0.601312637,-0.910558402), (-3.12368369,-0.700418531), (0.572659671,-1.817029), (0.362137467,1.34959018)], [(-0.956449389,-0.005283657), (-1.22591805,-5.20681238), (1.04414582,-0.218054503), (-1.45780349,0.0599878207)]]> : tensor<3x4xcomplex<f32>>
    %1 = stablehlo.constant dense<[[(0.442207187,-8.693150e-01), (2.89098954,-0.346056789)], [(5.42406416,0.265911907), (-0.613122045,-3.92265058)], [(1.57469404,6.20795536), (1.39309418,-2.15538692)], [(0.147095755,-3.99939346), (-0.128682762,-4.45214605)]]> : tensor<4x2xcomplex<f32>>
    return %0, %1 : tensor<3x4xcomplex<f32>>, tensor<4x2xcomplex<f32>>
  }
  func.func private @expected() -> tensor<3x2xcomplex<f32>> {
    %0 = stablehlo.constant dense<[[(9.7751789,-20.5735397), (18.2155914,6.08267784)], [(0.350143433,-6.1111555), (3.43432617,4.29050446)], [(-2.6690855,-15.7611561), (-21.0005112,12.2452965)]]> : tensor<3x2xcomplex<f32>>
    return %0 : tensor<3x2xcomplex<f32>>
  }
}
