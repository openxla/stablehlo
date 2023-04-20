// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x4xf32> {mhlo.sharding = ""}) -> tensor<4xf32> {
    %0 = stablehlo.constant dense<-2> : tensor<i64>
    %1 = stablehlo.add %arg0, %0 : tensor<i64>
    %2 = stablehlo.convert %arg0 : tensor<i64>
    %3 = stablehlo.constant dense<0> : tensor<i64>
    %4 = stablehlo.compare  LT, %1, %3,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
    %5 = stablehlo.add %1, %2 : tensor<i64>
    %6 = stablehlo.select %4, %5, %1 : tensor<i1>, tensor<i64>
    %7 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<i64>) -> tensor<1xi64>
    %8 = "stablehlo.gather"(%arg1, %7) {dimension_numbers = #stablehlo.gather<offset_dims = [0], collapsed_slice_dims = [0], start_index_map = [0]>, indices_are_sorted = true, slice_sizes = dense<[1, 4]> : tensor<2xi64>} : (tensor<?x4xf32>, tensor<1xi64>) -> tensor<4xf32>
    return %8 : tensor<4xf32>
  }
}

