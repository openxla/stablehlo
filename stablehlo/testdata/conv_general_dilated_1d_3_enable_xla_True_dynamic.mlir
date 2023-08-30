// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<5x?x16xf32> {mhlo.sharding = ""}, %arg2: tensor<4x16x16xf32> {mhlo.sharding = ""}) -> tensor<5x?x16xf32> {
    %0 = stablehlo.convolution(%arg1, %arg2) dim_numbers = [b, 0, f]x[0, i, o]->[b, 0, f], window = {pad = [[2, 2]], lhs_dilate = [2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<5x?x16xf32>, tensor<4x16x16xf32>) -> tensor<5x?x16xf32>
    return %0 : tensor<5x?x16xf32>
  }
}

