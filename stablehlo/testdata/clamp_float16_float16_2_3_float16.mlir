// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<2x3xf16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:3 = call @inputs() : () -> (tensor<f16>, tensor<2x3xf16>, tensor<f16>)
    %1 = call @expected() : () -> tensor<2x3xf16>
    %2 = stablehlo.broadcast_in_dim %0#0, dims = [] : (tensor<f16>) -> tensor<2x3xf16>
    %3 = stablehlo.broadcast_in_dim %0#2, dims = [] : (tensor<f16>) -> tensor<2x3xf16>
    %4 = stablehlo.clamp %2, %0#1, %3 : tensor<2x3xf16>
    stablehlo.custom_call @check.expect_close(%4, %1) {has_side_effect = true} : (tensor<2x3xf16>, tensor<2x3xf16>) -> ()
    return %4 : tensor<2x3xf16>
  }
  func.func private @inputs() -> (tensor<f16> {mhlo.layout_mode = "default"}, tensor<2x3xf16> {mhlo.layout_mode = "default"}, tensor<f16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[2.397460e-01, 2.158200e+00, 2.296880e+00], [-2.714840e+00, -2.687500e+00, -1.233400e+00]]> : tensor<2x3xf16>
    %cst_0 = stablehlo.constant dense<3.716800e+00> : tensor<f16>
    %cst_1 = stablehlo.constant dense<-4.324220e+00> : tensor<f16>
    return %cst_0, %cst, %cst_1 : tensor<f16>, tensor<2x3xf16>, tensor<f16>
  }
  func.func private @expected() -> (tensor<2x3xf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<-4.324220e+00> : tensor<2x3xf16>
    return %cst : tensor<2x3xf16>
  }
}
