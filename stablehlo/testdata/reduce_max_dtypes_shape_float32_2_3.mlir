// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<2x3xf32>
    %1 = call @expected() : () -> tensor<3xf32>
    %2 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %3 = stablehlo.reduce(%0 init: %2) across dimensions = [0] : (tensor<2x3xf32>, tensor<f32>) -> tensor<3xf32>
     reducer(%arg0: tensor<f32>, %arg1: tensor<f32>)  {
      %5 = stablehlo.maximum %arg0, %arg1 : tensor<f32>
      stablehlo.return %5 : tensor<f32>
    }
    %4 = stablehlo.custom_call @check.eq(%3, %1) : (tensor<3xf32>, tensor<3xf32>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> tensor<2x3xf32> {
    %0 = stablehlo.constant dense<[[1.33448267, 0.0435492285, 3.66509748], [0.677853763, 3.07493067, -3.59990954]]> : tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
  }
  func.func private @expected() -> tensor<3xf32> {
    %0 = stablehlo.constant dense<[1.33448267, 3.07493067, 3.66509748]> : tensor<3xf32>
    return %0 : tensor<3xf32>
  }
}
