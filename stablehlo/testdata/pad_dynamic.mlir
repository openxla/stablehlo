// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i64>, %arg1: tensor<?x2x5xf32> {mhlo.sharding = ""}, %arg2: tensor<f32> {mhlo.sharding = ""}) -> tensor<?x2x11xf32> {
    %0 = stablehlo.pad %arg1, %arg2, low = [0, 0, 1], high = [0, 0, 1], interior = [0, 0, 1] : (tensor<?x2x5xf32>, tensor<f32>) -> tensor<?x2x11xf32>
    return %0 : tensor<?x2x11xf32>
  }
}

