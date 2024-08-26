// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<2x3xbf16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:3 = call @inputs() : () -> (tensor<bf16>, tensor<2x3xbf16>, tensor<bf16>)
    %1 = call @expected() : () -> tensor<2x3xbf16>
    %2 = stablehlo.broadcast_in_dim %0#0, dims = [] : (tensor<bf16>) -> tensor<2x3xbf16>
    %3 = stablehlo.broadcast_in_dim %0#2, dims = [] : (tensor<bf16>) -> tensor<2x3xbf16>
    %4 = stablehlo.clamp %2, %0#1, %3 : tensor<2x3xbf16>
    stablehlo.custom_call @check.expect_close(%4, %1) {has_side_effect = true} : (tensor<2x3xbf16>, tensor<2x3xbf16>) -> ()
    return %4 : tensor<2x3xbf16>
  }
  func.func private @inputs() -> (tensor<bf16> {mhlo.layout_mode = "default"}, tensor<2x3xbf16> {mhlo.layout_mode = "default"}, tensor<bf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-4.406250e+00, -2.953130e+00, -1.609380e+00], [6.054690e-01, -1.748050e-01, 6.953130e-01]]> : tensor<2x3xbf16>
    %cst_0 = stablehlo.constant dense<-5.859380e-01> : tensor<bf16>
    %cst_1 = stablehlo.constant dense<-3.906250e+00> : tensor<bf16>
    return %cst_0, %cst, %cst_1 : tensor<bf16>, tensor<2x3xbf16>, tensor<bf16>
  }
  func.func private @expected() -> (tensor<2x3xbf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<-3.906250e+00> : tensor<2x3xbf16>
    return %cst : tensor<2x3xbf16>
  }
}
