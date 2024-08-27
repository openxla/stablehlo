// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<5x7xf32> {jax.result_info = "[0]", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<5x7xf32>
    %1 = call @expected() : () -> tensor<5x7xf32>
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
    }) : (tensor<5x7xf32>) -> tensor<5x7xf32>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<5x7xf32>, tensor<5x7xf32>) -> ()
    return %2 : tensor<5x7xf32>
  }
  func.func private @inputs() -> (tensor<5x7xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-1.07987475, 2.40121102, 0.609614312, 2.11366987, -3.75532937, -1.70798647, -1.60148156], [1.85626411, -0.232852742, 3.18949652, -2.20193195, -0.26601395, -3.44230628, -1.21295929], [-3.16205668, 0.570658684, -1.29143703, -1.20841086, 4.32051897, 1.61176097, -3.21153545], [0.805976331, 6.68937445, -1.94188154, -1.46865034, 0.651393652, 3.10745668, 9.86636352], [-0.469363391, 1.07057512, 4.19001722, 3.66513085, 2.18365979, -0.231317356, -0.3270244]]> : tensor<5x7xf32>
    return %cst : tensor<5x7xf32>
  }
  func.func private @expected() -> (tensor<5x7xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-3.16205668, -0.232852742, -1.94188154, -2.20193195, -3.75532937, -3.44230628, -3.21153545], [-1.07987475, 0.570658684, -1.29143703, -1.46865034, -0.26601395, -1.70798647, -1.60148156], [-0.469363391, 1.07057512, 0.609614312, -1.20841086, 0.651393652, -0.231317356, -1.21295929], [0.805976331, 2.40121102, 3.18949652, 2.11366987, 2.18365979, 1.61176097, -0.3270244], [1.85626411, 6.68937445, 4.19001722, 3.66513085, 4.32051897, 3.10745668, 9.86636352]]> : tensor<5x7xf32>
    return %cst : tensor<5x7xf32>
  }
}
