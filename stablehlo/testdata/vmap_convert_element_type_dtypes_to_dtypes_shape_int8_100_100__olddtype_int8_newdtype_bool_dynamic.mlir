// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x100x100xi8> {mhlo.sharding = ""}) -> tensor<?x100x100xi1> {
    %0 = stablehlo.constant dense<0> : tensor<i8>
    %1 = stablehlo.convert %arg0 : (tensor<i64>) -> tensor<i32>
    %2 = stablehlo.reshape %1 : (tensor<i32>) -> tensor<1xi32>
    %3 = stablehlo.constant dense<100> : tensor<1xi32>
    %4 = stablehlo.constant dense<100> : tensor<1xi32>
    %5 = stablehlo.concatenate %2, %3, %4, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %6 = stablehlo.dynamic_broadcast_in_dim %0, %5, dims = [] : (tensor<i8>, tensor<3xi32>) -> tensor<?x100x100xi8>
    %7 = stablehlo.compare  NE, %arg1, %6,  SIGNED : (tensor<?x100x100xi8>, tensor<?x100x100xi8>) -> tensor<?x100x100xi1>
    return %7 : tensor<?x100x100xi1>
  }
}

