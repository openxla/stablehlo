// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<15xf32>
    %1 = call @expected() : () -> tensor<ui8>
    %2 = call @argmin(%0) : (tensor<15xf32>) -> tensor<ui8>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<ui8>, tensor<ui8>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<15xf32> {
    %0 = stablehlo.constant dense<[-2.27708745, -0.500787795, -0.419249058, 5.6099987, 6.74828148, -1.33774209, 1.91096604, 0.737251222, 2.21453238, -0.531511664, 1.4323647, -1.17154801, -0.468607724, -2.13093019, -3.40978312]> : tensor<15xf32>
    return %0 : tensor<15xf32>
  }
  func.func private @expected() -> tensor<ui8> {
    %0 = stablehlo.constant dense<14> : tensor<ui8>
    return %0 : tensor<ui8>
  }
  func.func private @argmin(%arg0: tensor<15xf32>) -> tensor<ui8> {
    %0 = stablehlo.iota dim = 0 : tensor<15xui8>
    %1 = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %2 = stablehlo.constant dense<0> : tensor<ui8>
    %3:2 = stablehlo.reduce(%arg0 init: %1), (%0 init: %2) across dimensions = [0] : (tensor<15xf32>, tensor<15xui8>, tensor<f32>, tensor<ui8>) -> (tensor<f32>, tensor<ui8>)
     reducer(%arg1: tensor<f32>, %arg3: tensor<f32>) (%arg2: tensor<ui8>, %arg4: tensor<ui8>)  {
      %4 = stablehlo.compare  LT, %arg1, %arg3,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %5 = stablehlo.compare  NE, %arg1, %arg1,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %6 = stablehlo.or %4, %5 : tensor<i1>
      %7 = stablehlo.compare  EQ, %arg1, %arg3,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %8 = stablehlo.compare  LT, %arg2, %arg4,  UNSIGNED : (tensor<ui8>, tensor<ui8>) -> tensor<i1>
      %9 = stablehlo.and %7, %8 : tensor<i1>
      %10 = stablehlo.or %6, %9 : tensor<i1>
      %11 = stablehlo.select %6, %arg1, %arg3 : tensor<i1>, tensor<f32>
      %12 = stablehlo.select %10, %arg2, %arg4 : tensor<i1>, tensor<ui8>
      stablehlo.return %11, %12 : tensor<f32>, tensor<ui8>
    }
    return %3#1 : tensor<ui8>
  }
}
