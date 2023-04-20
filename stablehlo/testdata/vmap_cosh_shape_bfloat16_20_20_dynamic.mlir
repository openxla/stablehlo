// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x20x20xbf16> {mhlo.sharding = ""}) -> tensor<?x20x20xbf16> {
    %0 = stablehlo.constant dense<20> : tensor<1xi32>
    %1 = stablehlo.constant dense<5.000000e-01> : tensor<f32>
    %2 = stablehlo.convert %arg1 : (tensor<?x20x20xbf16>) -> tensor<?x20x20xf32>
    %3 = stablehlo.get_dimension_size %2, dim = 0 : (tensor<?x20x20xf32>) -> tensor<i32>
    %4 = stablehlo.reshape %3 : (tensor<i32>) -> tensor<1xi32>
    %5 = stablehlo.concatenate %4, %0, %0, dim = 0 : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %6 = stablehlo.dynamic_broadcast_in_dim %1, %5, dims = [] : (tensor<f32>, tensor<3xi32>) -> tensor<?x20x20xf32>
    %7 = stablehlo.log %6 : tensor<?x20x20xf32>
    %8 = stablehlo.add %2, %7 : tensor<?x20x20xf32>
    %9 = stablehlo.exponential %8 : tensor<?x20x20xf32>
    %10 = stablehlo.subtract %7, %2 : tensor<?x20x20xf32>
    %11 = stablehlo.exponential %10 : tensor<?x20x20xf32>
    %12 = stablehlo.add %9, %11 : tensor<?x20x20xf32>
    %13 = stablehlo.convert %12 : (tensor<?x20x20xf32>) -> tensor<?x20x20xbf16>
    return %13 : tensor<?x20x20xbf16>
  }
}

