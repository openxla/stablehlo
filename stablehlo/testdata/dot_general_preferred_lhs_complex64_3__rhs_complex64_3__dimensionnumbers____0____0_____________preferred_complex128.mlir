// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<3xcomplex<f32>>, tensor<3xcomplex<f32>>)
    %1 = call @expected() : () -> tensor<complex<f32>>
    %2 = "stablehlo.dot_general"(%0#0, %0#1) {dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [0], rhs_contracting_dimensions = [0]>} : (tensor<3xcomplex<f32>>, tensor<3xcomplex<f32>>) -> tensor<complex<f32>>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<complex<f32>>, tensor<complex<f32>>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<3xcomplex<f32>>, tensor<3xcomplex<f32>>) {
    %0 = stablehlo.constant dense<[(0.923159241,-1.0783174), (-0.396138638,1.18741775), (-0.968943417,-0.638475716)]> : tensor<3xcomplex<f32>>
    %1 = stablehlo.constant dense<[(-2.12202168,-3.80814266), (-3.00196218,-3.75992084), (0.360953718,0.632264376)]> : tensor<3xcomplex<f32>>
    return %0, %1 : tensor<3xcomplex<f32>>, tensor<3xcomplex<f32>>
  }
  func.func private @expected() -> tensor<complex<f32>> {
    %0 = stablehlo.constant dense<(-0.357618809,-4.1455307)> : tensor<complex<f32>>
    return %0 : tensor<complex<f32>>
  }
}

