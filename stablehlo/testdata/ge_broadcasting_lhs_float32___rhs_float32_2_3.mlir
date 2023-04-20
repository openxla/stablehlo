// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<f32>, tensor<2x3xf32>)
    %1 = call @expected() : () -> tensor<2x3xi1>
    %2 = stablehlo.broadcast_in_dim %0#0, dims = [] : (tensor<f32>) -> tensor<2x3xf32>
    %3 = stablehlo.compare  GE, %2, %0#1,  FLOAT : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xi1>
    %4 = stablehlo.custom_call @check.eq(%3, %1) : (tensor<2x3xi1>, tensor<2x3xi1>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<f32>, tensor<2x3xf32>) {
    %0 = stablehlo.constant dense<[[-0.723198115, 2.481570e+00, -1.35904646], [2.73648858, 3.69165468, 4.45880175]]> : tensor<2x3xf32>
    %1 = stablehlo.constant dense<1.37175632> : tensor<f32>
    return %1, %0 : tensor<f32>, tensor<2x3xf32>
  }
  func.func private @expected() -> tensor<2x3xi1> {
    %0 = stablehlo.constant dense<[[true, false, true], [false, false, false]]> : tensor<2x3xi1>
    return %0 : tensor<2x3xi1>
  }
}
