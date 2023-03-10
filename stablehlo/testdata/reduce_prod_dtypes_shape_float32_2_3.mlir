// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<2x3xf32>
    %1 = call @expected() : () -> tensor<3xf32>
    %2 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %3 = stablehlo.reduce(%0 init: %2) across dimensions = [0] : (tensor<2x3xf32>, tensor<f32>) -> tensor<3xf32>
     reducer(%arg0: tensor<f32>, %arg1: tensor<f32>)  {
      %5 = stablehlo.multiply %arg0, %arg1 : tensor<f32>
      stablehlo.return %5 : tensor<f32>
    }
    %4 = stablehlo.custom_call @check.eq(%3, %1) : (tensor<3xf32>, tensor<3xf32>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> tensor<2x3xf32> {
    %0 = stablehlo.constant dense<[[3.86633682, 1.91347504, -0.382918954], [-3.77680349, 0.86159414, -2.71516132]]> : tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
  }
  func.func private @expected() -> tensor<3xf32> {
    %0 = stablehlo.constant dense<[-14.6023941, 1.64863884, 1.03968668]> : tensor<3xf32>
    return %0 : tensor<3xf32>
  }
}
