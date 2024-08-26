// RUN-DISABLED(#2497): stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<5xf32> {jax.result_info = "[0]", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<5xf32>
    %1 = call @expected() : () -> tensor<5xf32>
    %2 = "stablehlo.sort"(%0) <{dimension = 0 : i64}> ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %3 = stablehlo.compare  EQ, %arg0, %cst,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %4 = stablehlo.select %3, %cst_0, %arg0 : tensor<i1>, tensor<f32>
      %5 = stablehlo.compare  NE, %arg0, %arg0,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %cst_1 = stablehlo.constant dense<0x7FC00000> : tensor<f32>
      %6 = stablehlo.select %5, %cst_1, %4 : tensor<i1>, tensor<f32>
      %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %7 = stablehlo.compare  EQ, %arg1, %cst_2,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %cst_3 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %8 = stablehlo.select %7, %cst_3, %arg1 : tensor<i1>, tensor<f32>
      %9 = stablehlo.compare  NE, %arg1, %arg1,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %cst_4 = stablehlo.constant dense<0x7FC00000> : tensor<f32>
      %10 = stablehlo.select %9, %cst_4, %8 : tensor<i1>, tensor<f32>
      %11 = stablehlo.compare  LT, %6, %10,  TOTALORDER : (tensor<f32>, tensor<f32>) -> tensor<i1>
      stablehlo.return %11 : tensor<i1>
    }) : (tensor<5xf32>) -> tensor<5xf32>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<5xf32>, tensor<5xf32>) -> ()
    return %2 : tensor<5xf32>
  }
  func.func private @inputs() -> (tensor<5xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[0x7F800000, 0x7FC00000, 0xFFC00000, 0xFF800000, 2.000000e+00]> : tensor<5xf32>
    return %cst : tensor<5xf32>
  }
  func.func private @expected() -> (tensor<5xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[0xFF800000, 2.000000e+00, 0x7F800000, 0xFFC00000, 0x7FC00000]> : tensor<5xf32>
    return %cst : tensor<5xf32>
  }
}
