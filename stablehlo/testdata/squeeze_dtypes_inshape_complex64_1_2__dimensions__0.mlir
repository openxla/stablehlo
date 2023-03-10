// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<1x2xcomplex<f32>>
    %1 = call @expected() : () -> tensor<2xcomplex<f32>>
    %2 = stablehlo.reshape %0 : (tensor<1x2xcomplex<f32>>) -> tensor<2xcomplex<f32>>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<2xcomplex<f32>>, tensor<2xcomplex<f32>>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<1x2xcomplex<f32>> {
    %0 = stablehlo.constant dense<[[(3.17430234,1.70979893), (-1.54939973,-1.52334642)]]> : tensor<1x2xcomplex<f32>>
    return %0 : tensor<1x2xcomplex<f32>>
  }
  func.func private @expected() -> tensor<2xcomplex<f32>> {
    %0 = stablehlo.constant dense<[(3.17430234,1.70979893), (-1.54939973,-1.52334642)]> : tensor<2xcomplex<f32>>
    return %0 : tensor<2xcomplex<f32>>
  }
}
