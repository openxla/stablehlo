// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:3 = call @inputs() : () -> (tensor<2x3xi1>, tensor<2x3xcomplex<f32>>, tensor<2x3xcomplex<f32>>)
    %1 = call @expected() : () -> tensor<2x3xcomplex<f32>>
    %2 = stablehlo.select %0#0, %0#2, %0#1 : tensor<2x3xi1>, tensor<2x3xcomplex<f32>>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<2x3xcomplex<f32>>, tensor<2x3xcomplex<f32>>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<2x3xi1>, tensor<2x3xcomplex<f32>>, tensor<2x3xcomplex<f32>>) {
    %0 = stablehlo.constant dense<true> : tensor<2x3xi1>
    %1 = stablehlo.constant dense<[[(-2.89732957,0.88831228), (0.310438275,3.24057746), (1.84898448,-2.39439654)], [(2.2174921,3.70856929), (-1.78715551,2.23971939), (1.7338953,-1.66348326)]]> : tensor<2x3xcomplex<f32>>
    %2 = stablehlo.constant dense<[[(-4.43330383,-2.2379992), (-1.95119584,1.03413486), (2.41349339,-3.18181968)], [(-0.051751636,-2.52063918), (1.8839823,1.2665329), (-4.7563343,-2.8792429)]]> : tensor<2x3xcomplex<f32>>
    return %0, %1, %2 : tensor<2x3xi1>, tensor<2x3xcomplex<f32>>, tensor<2x3xcomplex<f32>>
  }
  func.func private @expected() -> tensor<2x3xcomplex<f32>> {
    %0 = stablehlo.constant dense<[[(-4.43330383,-2.2379992), (-1.95119584,1.03413486), (2.41349339,-3.18181968)], [(-0.051751636,-2.52063918), (1.8839823,1.2665329), (-4.7563343,-2.8792429)]]> : tensor<2x3xcomplex<f32>>
    return %0 : tensor<2x3xcomplex<f32>>
  }
}
