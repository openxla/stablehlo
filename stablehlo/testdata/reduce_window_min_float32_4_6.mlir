// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3x5xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<4x6xf32>
    %1 = call @expected() : () -> tensor<3x5xf32>
    %cst = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<f32>
    %3 = "stablehlo.reduce_window"(%0, %2) <{window_dimensions = array<i64: 2, 2>}> ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %4 = stablehlo.minimum %arg0, %arg1 : tensor<f32>
      stablehlo.return %4 : tensor<f32>
    }) : (tensor<4x6xf32>, tensor<f32>) -> tensor<3x5xf32>
    stablehlo.custom_call @check.expect_close(%3, %1) {has_side_effect = true} : (tensor<3x5xf32>, tensor<3x5xf32>) -> ()
    return %3 : tensor<3x5xf32>
  }
  func.func private @inputs() -> (tensor<4x6xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-0.587396801, 1.41679704, 1.57286632, 2.75978231, -1.49650061, -2.30756736], [3.0710969, 4.19341516, 2.3716979, -3.34889412, 2.2374773, -1.73396564], [1.05131531, -0.597165704, 5.87361765, -0.149147883, -1.62176561, 0.108428128], [-2.76762557, 2.69764709, -1.18576527, -1.25375938, -2.87609529, 2.79383397]]> : tensor<4x6xf32>
    return %cst : tensor<4x6xf32>
  }
  func.func private @expected() -> (tensor<3x5xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-0.587396801, 1.41679704, -3.34889412, -3.34889412, -2.30756736], [-0.597165704, -0.597165704, -3.34889412, -3.34889412, -1.73396564], [-2.76762557, -1.18576527, -1.25375938, -2.87609529, -2.87609529]]> : tensor<3x5xf32>
    return %cst : tensor<3x5xf32>
  }
}
