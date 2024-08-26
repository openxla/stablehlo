// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<1xbf16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<3xbf16>
    %1 = call @expected() : () -> tensor<1xbf16>
    %2 = stablehlo.slice %0 [1:2] : (tensor<3xbf16>) -> tensor<1xbf16>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<1xbf16>, tensor<1xbf16>) -> ()
    return %2 : tensor<1xbf16>
  }
  func.func private @inputs() -> (tensor<3xbf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[-1.960940e+00, -3.300780e-01, 8.945310e-01]> : tensor<3xbf16>
    return %cst : tensor<3xbf16>
  }
  func.func private @expected() -> (tensor<1xbf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<-3.300780e-01> : tensor<1xbf16>
    return %cst : tensor<1xbf16>
  }
}
