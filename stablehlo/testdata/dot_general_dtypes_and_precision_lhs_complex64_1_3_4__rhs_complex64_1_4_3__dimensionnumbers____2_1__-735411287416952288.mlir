// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<1x3x4xcomplex<f32>>, tensor<1x4x3xcomplex<f32>>)
    %1 = call @expected() : () -> tensor<1xcomplex<f32>>
    %2 = "stablehlo.dot_general"(%0#0, %0#1) {dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2, 1], rhs_contracting_dimensions = [1, 2]>, precision_config = [#stablehlo<precision HIGHEST>, #stablehlo<precision HIGHEST>]} : (tensor<1x3x4xcomplex<f32>>, tensor<1x4x3xcomplex<f32>>) -> tensor<1xcomplex<f32>>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<1xcomplex<f32>>, tensor<1xcomplex<f32>>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<1x3x4xcomplex<f32>>, tensor<1x4x3xcomplex<f32>>) {
    %0 = stablehlo.constant dense<[[[(3.68449879,0.768986701), (1.78898776,6.53381538), (2.21337438,5.1064415), (5.74728584,-0.386461824)], [(6.23531103,1.18506539), (-5.77455282,0.560406089), (1.85820413,0.414159298), (-0.749784052,-0.725378513)], [(-1.25899792,1.02634156), (3.78332114,-1.85186446), (2.25832248,2.15603304), (-1.259840e+00,-0.225763828)]]]> : tensor<1x3x4xcomplex<f32>>
    %1 = stablehlo.constant dense<[[[(-1.24749029,0.520909429), (-5.68658304,-2.32884407), (0.857682883,-2.61452222)], [(-4.36643267,3.55293798), (-0.710062623,1.66516089), (-0.477060288,-1.42167974)], [(-5.03311443,-1.04349339), (1.91064668,1.69355142), (0.853816688,2.54109859)], [(2.05972171,1.08951521), (4.37291431,0.949390828), (3.171720e+00,2.22521091)]]]> : tensor<1x4x3xcomplex<f32>>
    return %0, %1 : tensor<1x3x4xcomplex<f32>>, tensor<1x4x3xcomplex<f32>>
  }
  func.func private @expected() -> tensor<1xcomplex<f32>> {
    %0 = stablehlo.constant dense<(-68.7252426,-71.2408676)> : tensor<1xcomplex<f32>>
    return %0 : tensor<1xcomplex<f32>>
  }
}
