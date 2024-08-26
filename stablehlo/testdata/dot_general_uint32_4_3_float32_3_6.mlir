// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x6xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<4x3xui32>, tensor<3x6xf32>)
    %1 = call @expected() : () -> tensor<4x6xf32>
    %2 = stablehlo.convert %0#0 : (tensor<4x3xui32>) -> tensor<4x3xf32>
    %3 = stablehlo.convert %0#1 : tensor<3x6xf32>
    %4 = stablehlo.dot_general %2, %3, contracting_dims = [1] x [0] : (tensor<4x3xf32>, tensor<3x6xf32>) -> tensor<4x6xf32>
    stablehlo.custom_call @check.expect_almost_eq(%4, %1) {has_side_effect = true} : (tensor<4x6xf32>, tensor<4x6xf32>) -> ()
    return %4 : tensor<4x6xf32>
  }
  func.func private @inputs() -> (tensor<4x3xui32> {mhlo.layout_mode = "default"}, tensor<3x6xf32> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[2, 0, 3], [0, 2, 0], [3, 0, 0], [3, 3, 5]]> : tensor<4x3xui32>
    %cst = stablehlo.constant dense<[[4.15925264, 3.60511088, 1.3781395, 1.36220229, 5.20291901, 1.18606138], [2.79948449, -6.24050951, 1.52686501, -3.68102241, 0.540223777, -1.89685023], [5.18335104, -2.77697968, 3.13168025, 0.119321391, -2.42349887, -1.81318474]]> : tensor<3x6xf32>
    return %c, %cst : tensor<4x3xui32>, tensor<3x6xf32>
  }
  func.func private @expected() -> (tensor<4x6xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[23.8685589, -1.12071729, 12.1513195, 3.08236885, 3.13534141, -3.06743145], [5.59896898, -12.481019, 3.053730e+00, -7.36204481, 1.08044755, -3.79370046], [12.4777584, 10.8153324, 4.13441849, 4.08660698, 15.608757, 3.55818415], [46.7929649, -21.7910938, 24.373415, -6.35985327, 5.11193466, -11.1982899]]> : tensor<4x6xf32>
    return %cst : tensor<4x6xf32>
  }
}
