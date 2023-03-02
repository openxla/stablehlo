// RUN: diff <(stablehlo-opt %s.0_9_0.bc --vhlo-to-version=target=current --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-opt %s --stablehlo-legalize-to-vhlo --vhlo-to-version=target=current -emit-bytecode | stablehlo-opt --vhlo-legalize-to-stablehlo) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x4xf32> {mhlo.sharding = ""}) -> tensor<4xf32> {
    %0 = stablehlo.constant dense<1> : tensor<i64>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<i64>) -> tensor<1xi64>
    %2 = "stablehlo.gather"(%arg1, %1) {dimension_numbers = #stablehlo.gather<offset_dims = [0], collapsed_slice_dims = [0], start_index_map = [0]>, indices_are_sorted = true, slice_sizes = dense<[1, 4]> : tensor<2xi64>} : (tensor<?x4xf32>, tensor<1xi64>) -> tensor<4xf32>
    return %2 : tensor<4xf32>
  }
}

