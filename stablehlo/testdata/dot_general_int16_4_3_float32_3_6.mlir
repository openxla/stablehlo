// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x6xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<4x3xi16>, tensor<3x6xf32>)
    %1 = call @expected() : () -> tensor<4x6xf32>
    %2 = stablehlo.convert %0#0 : (tensor<4x3xi16>) -> tensor<4x3xf32>
    %3 = stablehlo.convert %0#1 : tensor<3x6xf32>
    %4 = stablehlo.dot_general %2, %3, contracting_dims = [1] x [0] : (tensor<4x3xf32>, tensor<3x6xf32>) -> tensor<4x6xf32>
    stablehlo.custom_call @check.expect_close(%4, %1) {has_side_effect = true} : (tensor<4x6xf32>, tensor<4x6xf32>) -> ()
    return %4 : tensor<4x6xf32>
  }
  func.func private @inputs() -> (tensor<4x3xi16> {mhlo.layout_mode = "default"}, tensor<3x6xf32> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[5, 2, -2], [1, 0, -1], [-5, 2, -2], [-4, -1, -5]]> : tensor<4x3xi16>
    %cst = stablehlo.constant dense<[[-2.94647026, 0.734722912, -2.58595443, 1.07965136, 1.0958364, 2.70218849], [-3.40507245, -1.9028939, 7.62151289, 3.63030744, -0.233068526, -1.80128574], [2.33670807, 1.56321156, -0.354040384, 1.42250979, -4.28100061, 1.82065916]]> : tensor<3x6xf32>
    return %c, %cst : tensor<4x3xi16>, tensor<3x6xf32>
  }
  func.func private @expected() -> (tensor<4x6xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-26.2159119, -3.25859642, 3.02133417, 9.81385231, 13.5750465, 6.26705265], [-5.28317833, -0.828488647, -2.23191404, -0.342858434, 5.37683678, 0.881529331], [3.24879026, -10.6058254, 28.8808784, -0.982661485, 2.61668205, -20.7548332], [3.50741291, -8.85205554, 4.49250698, -15.0614614, 17.2547264, -18.1107635]]> : tensor<4x6xf32>
    return %cst : tensor<4x6xf32>
  }
}
