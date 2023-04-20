// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x3x9x10xf32> {mhlo.sharding = ""}, %arg2: tensor<3x3x4x5xf32> {mhlo.sharding = ""}) -> tensor<?x3x3x1xf32> {
    %0 = stablehlo.convolution(%arg1, %arg2) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {stride = [2, 3], rhs_dilate = [1, 2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<?x3x9x10xf32>, tensor<3x3x4x5xf32>) -> tensor<?x3x3x1xf32>
    return %0 : tensor<?x3x3x1xf32>
  }
}

