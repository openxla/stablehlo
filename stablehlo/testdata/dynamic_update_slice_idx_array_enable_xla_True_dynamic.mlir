// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x4xf32> {mhlo.sharding = ""}, %arg2: tensor<2xi32> {mhlo.sharding = ""}) -> tensor<?x4xf32> {
    %0 = "stablehlo.slice"(%arg2) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xi32>) -> tensor<1xi32>
    %1 = stablehlo.reshape %0 : (tensor<1xi32>) -> tensor<i32>
    %2 = "stablehlo.slice"(%arg2) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xi32>) -> tensor<1xi32>
    %3 = stablehlo.reshape %2 : (tensor<1xi32>) -> tensor<i32>
    %4 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %5 = stablehlo.constant dense<0> : tensor<i32>
    %6 = stablehlo.compare  LT, %1, %5,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %7 = stablehlo.add %1, %4 : tensor<i32>
    %8 = stablehlo.select %6, %7, %1 : tensor<i1>, tensor<i32>
    %9 = stablehlo.constant dense<0> : tensor<i32>
    %10 = stablehlo.compare  LT, %3, %9,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %11 = stablehlo.constant dense<4> : tensor<i32>
    %12 = stablehlo.add %3, %11 : tensor<i32>
    %13 = stablehlo.select %10, %12, %3 : tensor<i1>, tensor<i32>
    %14 = stablehlo.dynamic_update_slice %arg1, %arg1, %8, %13 : (tensor<?x4xf32>, tensor<?x4xf32>, tensor<i32>, tensor<i32>) -> tensor<?x4xf32>
    return %14 : tensor<?x4xf32>
  }
}

