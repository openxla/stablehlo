// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<ui64> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<15xf32>
    %1 = call @expected() : () -> tensor<ui64>
    %2 = call @argmax(%0) : (tensor<15xf32>) -> tensor<ui64>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<ui64>, tensor<ui64>) -> ()
    return %2 : tensor<ui64>
  }
  func.func private @inputs() -> (tensor<15xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[-3.43464923, 7.15940142, 2.93741393, -0.346413374, 1.37187719, -1.83673358, 5.99954605, 2.46723032, -2.66364694, 0.517188668, -5.9878273, 6.03587484, 2.35249352, 1.32477033, 2.85481739]> : tensor<15xf32>
    return %cst : tensor<15xf32>
  }
  func.func private @expected() -> (tensor<ui64> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<1> : tensor<ui64>
    return %c : tensor<ui64>
  }
  func.func private @argmax(%arg0: tensor<15xf32>) -> tensor<ui64> {
    %0 = stablehlo.iota dim = 0 : tensor<15xui64>
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %c = stablehlo.constant dense<0> : tensor<ui64>
    %1:2 = stablehlo.reduce(%arg0 init: %cst), (%0 init: %c) across dimensions = [0] : (tensor<15xf32>, tensor<15xui64>, tensor<f32>, tensor<ui64>) -> (tensor<f32>, tensor<ui64>)
     reducer(%arg1: tensor<f32>, %arg3: tensor<f32>) (%arg2: tensor<ui64>, %arg4: tensor<ui64>)  {
      %2 = stablehlo.compare  GT, %arg1, %arg3,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %3 = stablehlo.compare  NE, %arg1, %arg1,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %4 = stablehlo.or %2, %3 : tensor<i1>
      %5 = stablehlo.compare  EQ, %arg1, %arg3,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %6 = stablehlo.compare  LT, %arg2, %arg4,  UNSIGNED : (tensor<ui64>, tensor<ui64>) -> tensor<i1>
      %7 = stablehlo.and %5, %6 : tensor<i1>
      %8 = stablehlo.or %4, %7 : tensor<i1>
      %9 = stablehlo.select %4, %arg1, %arg3 : tensor<i1>, tensor<f32>
      %10 = stablehlo.select %8, %arg2, %arg4 : tensor<i1>, tensor<ui64>
      stablehlo.return %9, %10 : tensor<f32>, tensor<ui64>
    }
    return %1#1 : tensor<ui64>
  }
}
