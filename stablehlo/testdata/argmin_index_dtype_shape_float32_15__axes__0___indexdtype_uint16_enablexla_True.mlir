// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<15xf32>
    %1 = call @expected() : () -> tensor<ui16>
    %2 = call @argmin(%0) : (tensor<15xf32>) -> tensor<ui16>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<ui16>, tensor<ui16>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<15xf32> {
    %0 = stablehlo.constant dense<[1.04617405, 2.1091435, -1.1197598, 1.45735288, -0.62860161, 2.01927543, 4.47722721, -1.191540e+00, -0.0859221816, -5.81972408, -3.88508272, 1.68279612, 1.18479204, -3.28509545, 2.10967135]> : tensor<15xf32>
    return %0 : tensor<15xf32>
  }
  func.func private @expected() -> tensor<ui16> {
    %0 = stablehlo.constant dense<9> : tensor<ui16>
    return %0 : tensor<ui16>
  }
  func.func private @argmin(%arg0: tensor<15xf32>) -> tensor<ui16> {
    %0 = stablehlo.iota dim = 0 : tensor<15xui16>
    %1 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %2 = stablehlo.constant dense<0> : tensor<ui16>
    %3:2 = stablehlo.reduce(%arg0 init: %1), (%0 init: %2) across dimensions = [0] : (tensor<15xf32>, tensor<15xui16>, tensor<f32>, tensor<ui16>) -> (tensor<f32>, tensor<ui16>)
     reducer(%arg1: tensor<f32>, %arg3: tensor<f32>) (%arg2: tensor<ui16>, %arg4: tensor<ui16>)  {
      %4 = stablehlo.compare  LT, %arg1, %arg3,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
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
