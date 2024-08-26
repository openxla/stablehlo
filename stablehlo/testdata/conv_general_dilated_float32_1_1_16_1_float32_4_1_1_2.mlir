// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<1x1x16x2xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<1x1x16x1xf32>, tensor<4x1x1x2xf32>)
    %1 = call @expected() : () -> tensor<1x1x16x2xf32>
    %2 = stablehlo.convolution(%0#0, %0#1) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {pad = [[1, 2], [0, 0]]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x1x16x1xf32>, tensor<4x1x1x2xf32>) -> tensor<1x1x16x2xf32>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<1x1x16x2xf32>, tensor<1x1x16x2xf32>) -> ()
    return %2 : tensor<1x1x16x2xf32>
  }
  func.func private @inputs() -> (tensor<1x1x16x1xf32> {mhlo.layout_mode = "default"}, tensor<4x1x1x2xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[[[0.249547243], [-3.84318519], [-1.0406214], [1.16896808], [-0.0966261476], [0.145545349], [-0.652937531], [4.2463026], [-1.61683857], [-2.21878624], [-6.34280634], [-1.90672576], [-4.64762354], [3.25072265], [-0.142132208], [-0.168175161]]]]> : tensor<1x1x16x1xf32>
    %cst_0 = stablehlo.constant dense<[[[[0.354699761, 0.679385543]]], [[[0.609961211, 0.152189493]]], [[[-2.194420e+00, -2.72838569]]], [[[-0.477422208, -1.72463417]]]]> : tensor<4x1x1x2xf32>
    return %cst, %cst_0 : tensor<1x1x16x1xf32>, tensor<4x1x1x2xf32>
  }
  func.func private @expected() -> (tensor<1x1x16x2xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[[[0.15221414, 0.0379784666], [-2.34419394, -0.584892392], [-0.634738684, -0.158371642], [0.713025212, 0.177904665], [-5.893820e-02, -0.0147054847], [0.0887770206, 0.0221504737], [-0.398266554, -0.0993702337], [2.59007978, 0.646242618], [-0.986208796, -0.24606584], [-1.35337353, -0.337675959], [-3.86886573, -0.965308487], [-1.16302872, -0.290183634], [-2.834870e+00, -0.707319498], [1.98281467, 0.494725823], [-0.0866951346, -0.0216310285], [-0.102580324, -0.0255944934]]]]> : tensor<1x1x16x2xf32>
    return %cst : tensor<1x1x16x2xf32>
  }
}
