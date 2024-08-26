// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3xf16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<2x3xf16>
    %1 = call @expected() : () -> tensor<3xf16>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f16>
    %2 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<2x3xf16>, tensor<f16>) -> tensor<3xf16>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<3xf16>, tensor<3xf16>) -> ()
    return %2 : tensor<3xf16>
  }
  func.func private @inputs() -> (tensor<2x3xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[1.444340e+00, 1.369140e+00, -4.031250e+00], [1.280270e+00, -1.709960e+00, 2.041020e+00]]> : tensor<2x3xf16>
    return %cst : tensor<2x3xf16>
  }
  func.func private @expected() -> (tensor<3xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[2.724610e+00, -3.408200e-01, -1.990230e+00]> : tensor<3xf16>
    return %cst : tensor<3xf16>
  }
}
