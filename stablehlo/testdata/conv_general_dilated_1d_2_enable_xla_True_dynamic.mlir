// RUN: diff <(stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-opt %s --stablehlo-legalize-to-vhlo --vhlo-to-version=target=current -emit-bytecode | stablehlo-opt --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x12x16xf32> {mhlo.sharding = ""}, %arg2: tensor<4x16x16xf32> {mhlo.sharding = ""}) -> tensor<?x24x16xf32> {
    %0 = stablehlo.convolution(%arg1, %arg2) dim_numbers = [b, 0, f]x[0, i, o]->[b, 0, f], window = {pad = [[2, 2]], lhs_dilate = [2]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<?x12x16xf32>, tensor<4x16x16xf32>) -> tensor<?x24x16xf32>
    return %0 : tensor<?x24x16xf32>
  }
}

