// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<1x3x4xcomplex<f32>>, tensor<1x4x3xcomplex<f32>>)
    %1 = call @expected() : () -> tensor<1xcomplex<f32>>
    %2 = "stablehlo.dot_general"(%0#0, %0#1) {dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2, 1], rhs_contracting_dimensions = [1, 2]>} : (tensor<1x3x4xcomplex<f32>>, tensor<1x4x3xcomplex<f32>>) -> tensor<1xcomplex<f32>>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<1xcomplex<f32>>, tensor<1xcomplex<f32>>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<1x3x4xcomplex<f32>>, tensor<1x4x3xcomplex<f32>>) {
    %0 = stablehlo.constant dense<[[[(2.71014166,-5.58421946), (3.05123162,-4.1052165), (-0.0548705719,-1.44886839), (4.46283388,1.10272384)], [(-2.97033715,-0.0108122807), (-0.233915508,2.22155595), (1.00114608,0.583914459), (-1.32969403,-2.75252056)], [(-2.90731359,-4.28674126), (-2.48315525,4.85155344), (-1.76463568,2.6458106), (3.76505256,1.76979804)]]]> : tensor<1x3x4xcomplex<f32>>
    %1 = stablehlo.constant dense<[[[(-2.53046203,-4.23012114), (-0.151592821,-0.350648701), (3.48485065,6.23519229)], [(-2.980490e+00,2.12334871), (-3.41384196,-4.24195385), (-1.14971662,5.98359394)], [(-3.49577355,-1.77865875), (-1.92902601,-2.01291609), (-0.251457125,0.764541566)], [(-2.31303549,-3.34804082), (-0.98489362,-1.30129862), (1.36343217,2.5126946)]]]> : tensor<1x4x3xcomplex<f32>>
    return %0, %1 : tensor<1x3x4xcomplex<f32>>, tensor<1x4x3xcomplex<f32>>
  }
  func.func private @expected() -> tensor<1xcomplex<f32>> {
    %0 = stablehlo.constant dense<(-42.7028503,-38.8414307)> : tensor<1xcomplex<f32>>
    return %0 : tensor<1xcomplex<f32>>
  }
}

