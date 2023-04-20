// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x20x20xbf16> {mhlo.sharding = ""}) -> tensor<?x20x20xbf16> {
    %0 = stablehlo.constant dense<20> : tensor<1xi32>
    %1 = stablehlo.constant dense<5.000000e-01> : tensor<bf16>
    %2 = stablehlo.constant dense<0x7FC0> : tensor<bf16>
    %3 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %4 = stablehlo.abs %arg1 : tensor<?x20x20xbf16>
    %5 = stablehlo.get_dimension_size %arg1, dim = 0 : (tensor<?x20x20xbf16>) -> tensor<i32>
    %6 = stablehlo.reshape %5 : (tensor<i32>) -> tensor<1xi32>
    %7 = stablehlo.concatenate %6, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %8 = stablehlo.dynamic_broadcast_in_dim %3, %7, dims = [] : (tensor<bf16>, tensor<3xi32>) -> tensor<?x20x20xbf16>
    %9 = stablehlo.compare  GT, %4, %8 : (tensor<?x20x20xbf16>, tensor<?x20x20xbf16>) -> tensor<?x20x20xi1>
    %10 = stablehlo.get_dimension_size %arg1, dim = 0 : (tensor<?x20x20xbf16>) -> tensor<i32>
    %11 = stablehlo.reshape %10 : (tensor<i32>) -> tensor<1xi32>
    %12 = stablehlo.concatenate %11, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %13 = stablehlo.dynamic_broadcast_in_dim %2, %12, dims = [] : (tensor<bf16>, tensor<3xi32>) -> tensor<?x20x20xbf16>
    %14 = stablehlo.log_plus_one %arg1 : tensor<?x20x20xbf16>
    %15 = stablehlo.negate %arg1 : tensor<?x20x20xbf16>
    %16 = stablehlo.log_plus_one %15 : tensor<?x20x20xbf16>
    %17 = stablehlo.subtract %14, %16 : tensor<?x20x20xbf16>
    %18 = stablehlo.get_dimension_size %arg1, dim = 0 : (tensor<?x20x20xbf16>) -> tensor<i32>
    %19 = stablehlo.reshape %18 : (tensor<i32>) -> tensor<1xi32>
    %20 = stablehlo.concatenate %19, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %21 = stablehlo.dynamic_broadcast_in_dim %1, %20, dims = [] : (tensor<bf16>, tensor<3xi32>) -> tensor<?x20x20xbf16>
    %22 = stablehlo.multiply %17, %21 : tensor<?x20x20xbf16>
    %23 = stablehlo.select %9, %13, %22 : tensor<?x20x20xi1>, tensor<?x20x20xbf16>
    return %23 : tensor<?x20x20xbf16>
  }
}

