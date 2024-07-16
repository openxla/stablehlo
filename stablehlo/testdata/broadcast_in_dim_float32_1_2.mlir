// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x3x2xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<1x2xf32>
    %1 = call @expected() : () -> tensor<4x3x2xf32>
    %2 = stablehlo.broadcast_in_dim %0, dims = [0, 2] : (tensor<1x2xf32>) -> tensor<4x3x2xf32>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<4x3x2xf32>, tensor<4x3x2xf32>) -> ()
    return %2 : tensor<4x3x2xf32>
  }
  func.func private @inputs() -> (tensor<1x2xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[0.544339418, 2.04886508]]> : tensor<1x2xf32>
    return %cst : tensor<1x2xf32>
  }
  func.func private @expected() -> (tensor<4x3x2xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[[0.544339418, 2.04886508], [0.544339418, 2.04886508], [0.544339418, 2.04886508]], [[0.544339418, 2.04886508], [0.544339418, 2.04886508], [0.544339418, 2.04886508]], [[0.544339418, 2.04886508], [0.544339418, 2.04886508], [0.544339418, 2.04886508]], [[0.544339418, 2.04886508], [0.544339418, 2.04886508], [0.544339418, 2.04886508]]]> : tensor<4x3x2xf32>
    return %cst : tensor<4x3x2xf32>
  }
}
