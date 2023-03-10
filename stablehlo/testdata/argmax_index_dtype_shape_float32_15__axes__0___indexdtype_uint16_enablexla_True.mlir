// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<15xf32>
    %1 = call @expected() : () -> tensor<ui16>
    %2 = call @argmax(%0) : (tensor<15xf32>) -> tensor<ui16>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<ui16>, tensor<ui16>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<15xf32> {
    %0 = stablehlo.constant dense<[-1.85265541, -5.7855382, 2.67888737, -6.73458242, 0.673333704, 4.5620203, -2.67249727, -0.597331285, 3.63108253, 0.862780272, -1.37556636, -1.68591523, -3.270320e-02, -2.07354689, -5.69280529]> : tensor<15xf32>
    return %0 : tensor<15xf32>
  }
  func.func private @expected() -> tensor<ui16> {
    %0 = stablehlo.constant dense<5> : tensor<ui16>
    return %0 : tensor<ui16>
  }
  func.func private @argmax(%arg0: tensor<15xf32>) -> tensor<ui16> {
    %0 = stablehlo.iota dim = 0 : tensor<15xui16>
    %1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %2 = stablehlo.constant dense<0> : tensor<ui16>
    %3:2 = stablehlo.reduce(%arg0 init: %1), (%0 init: %2) across dimensions = [0] : (tensor<15xf32>, tensor<15xui16>, tensor<f32>, tensor<ui16>) -> (tensor<f32>, tensor<ui16>)
     reducer(%arg1: tensor<f32>, %arg3: tensor<f32>) (%arg2: tensor<ui16>, %arg4: tensor<ui16>)  {
      %4 = stablehlo.compare  GT, %arg1, %arg3,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %5 = stablehlo.compare  NE, %arg1, %arg1,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %6 = stablehlo.or %4, %5 : tensor<i1>
      %7 = stablehlo.compare  EQ, %arg1, %arg3,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %8 = stablehlo.compare  LT, %arg2, %arg4,  UNSIGNED : (tensor<ui16>, tensor<ui16>) -> tensor<i1>
      %9 = stablehlo.and %7, %8 : tensor<i1>
      %10 = stablehlo.or %6, %9 : tensor<i1>
      %11 = stablehlo.select %6, %arg1, %arg3 : tensor<i1>, tensor<f32>
      %12 = stablehlo.select %10, %arg2, %arg4 : tensor<i1>, tensor<ui16>
      stablehlo.return %11, %12 : tensor<f32>, tensor<ui16>
    }
    return %3#1 : tensor<ui16>
  }
}
