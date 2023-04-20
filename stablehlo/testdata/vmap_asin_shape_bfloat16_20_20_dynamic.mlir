// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x20x20xbf16> {mhlo.sharding = ""}) -> tensor<?x20x20xbf16> {
    %0 = stablehlo.constant dense<20> : tensor<1xi32>
    %1 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %2 = stablehlo.constant dense<2.000000e+00> : tensor<bf16>
    %3 = stablehlo.get_dimension_size %arg1, dim = 0 : (tensor<?x20x20xbf16>) -> tensor<i32>
    %4 = stablehlo.reshape %3 : (tensor<i32>) -> tensor<1xi32>
    %5 = stablehlo.concatenate %4, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %6 = stablehlo.dynamic_broadcast_in_dim %2, %5, dims = [] : (tensor<bf16>, tensor<3xi32>) -> tensor<?x20x20xbf16>
    %7 = stablehlo.get_dimension_size %arg1, dim = 0 : (tensor<?x20x20xbf16>) -> tensor<i32>
    %8 = stablehlo.reshape %7 : (tensor<i32>) -> tensor<1xi32>
    %9 = stablehlo.concatenate %8, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %10 = stablehlo.dynamic_broadcast_in_dim %1, %9, dims = [] : (tensor<bf16>, tensor<3xi32>) -> tensor<?x20x20xbf16>
    %11 = stablehlo.get_dimension_size %arg1, dim = 0 : (tensor<?x20x20xbf16>) -> tensor<i32>
    %12 = stablehlo.reshape %11 : (tensor<i32>) -> tensor<1xi32>
    %13 = stablehlo.concatenate %12, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %14 = stablehlo.dynamic_broadcast_in_dim %1, %13, dims = [] : (tensor<bf16>, tensor<3xi32>) -> tensor<?x20x20xbf16>
    %15 = stablehlo.multiply %arg1, %arg1 : tensor<?x20x20xbf16>
    %16 = stablehlo.subtract %14, %15 : tensor<?x20x20xbf16>
    %17 = stablehlo.sqrt %16 : tensor<?x20x20xbf16>
    %18 = stablehlo.add %10, %17 : tensor<?x20x20xbf16>
    %19 = stablehlo.atan2 %arg1, %18 : tensor<?x20x20xbf16>
    %20 = stablehlo.multiply %6, %19 : tensor<?x20x20xbf16>
    return %20 : tensor<?x20x20xbf16>
  }
}

