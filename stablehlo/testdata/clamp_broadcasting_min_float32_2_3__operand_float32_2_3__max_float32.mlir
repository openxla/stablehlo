// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:3 = call @inputs() : () -> (tensor<2x3xf32>, tensor<2x3xf32>, tensor<f32>)
    %1 = call @expected() : () -> tensor<2x3xf32>
    %2 = stablehlo.broadcast_in_dim %0#2, dims = [] : (tensor<f32>) -> tensor<2x3xf32>
    %3 = stablehlo.clamp %0#0, %0#1, %2 : tensor<2x3xf32>
    %4 = stablehlo.custom_call @check.eq(%3, %1) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<2x3xf32>, tensor<2x3xf32>, tensor<f32>) {
    %0 = stablehlo.constant dense<[[-0.422537923, -0.570146441, 5.37270308], [2.03217745, -5.92806864, -0.280902356]]> : tensor<2x3xf32>
    %1 = stablehlo.constant dense<[[3.49295878, -0.928503811, -0.57337445], [0.804368913, -4.49679565, 0.416748792]]> : tensor<2x3xf32>
    %2 = stablehlo.constant dense<1.73386252> : tensor<f32>
    return %0, %1, %2 : tensor<2x3xf32>, tensor<2x3xf32>, tensor<f32>
  }
  func.func private @expected() -> tensor<2x3xf32> {
    %0 = stablehlo.constant dense<[[1.73386252, -0.570146441, 1.73386252], [1.73386252, -4.49679565, 0.416748792]]> : tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
  }
}
