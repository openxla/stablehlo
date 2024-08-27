// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<2x3xbf16> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:4 = call @inputs() : () -> (tensor<2x3xi32>, tensor<2x3xbf16>, tensor<2x3xbf16>, tensor<2x3xbf16>)
    %1 = call @expected() : () -> tensor<2x3xbf16>
    %c = stablehlo.constant dense<1> : tensor<i32>
    %2 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2x3xi32>
    %3 = stablehlo.compare  LT, %0#0, %2,  SIGNED : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi1>
    %c_0 = stablehlo.constant dense<2> : tensor<i32>
    %4 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<2x3xi32>
    %5 = stablehlo.compare  LT, %0#0, %4,  SIGNED : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi1>
    %6 = stablehlo.select %5, %0#2, %0#3 : tensor<2x3xi1>, tensor<2x3xbf16>
    %7 = stablehlo.select %3, %0#1, %6 : tensor<2x3xi1>, tensor<2x3xbf16>
    stablehlo.custom_call @check.expect_close(%7, %1) {has_side_effect = true} : (tensor<2x3xbf16>, tensor<2x3xbf16>) -> ()
    return %7 : tensor<2x3xbf16>
  }
  func.func private @inputs() -> (tensor<2x3xi32> {mhlo.layout_mode = "default"}, tensor<2x3xbf16> {mhlo.layout_mode = "default"}, tensor<2x3xbf16> {mhlo.layout_mode = "default"}, tensor<2x3xbf16> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[0, 2, 1], [2, 0, 1]]> : tensor<2x3xi32>
    %cst = stablehlo.constant dense<[[-1.570310e+00, 2.243040e-03, -5.078130e-01], [-2.375000e+00, -1.609380e+00, -4.187500e+00]]> : tensor<2x3xbf16>
    %cst_0 = stablehlo.constant dense<[[-4.687500e+00, 3.156250e+00, -2.890630e+00], [2.687500e+00, 4.562500e+00, 3.296880e+00]]> : tensor<2x3xbf16>
    %cst_1 = stablehlo.constant dense<[[1.742190e+00, -2.640630e+00, 1.164060e+00], [-2.171880e+00, 2.265630e-01, 3.734380e+00]]> : tensor<2x3xbf16>
    return %c, %cst, %cst_0, %cst_1 : tensor<2x3xi32>, tensor<2x3xbf16>, tensor<2x3xbf16>, tensor<2x3xbf16>
  }
  func.func private @expected() -> (tensor<2x3xbf16> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-1.570310e+00, -2.640630e+00, -2.890630e+00], [-2.171880e+00, -1.609380e+00, 3.296880e+00]]> : tensor<2x3xbf16>
    return %cst : tensor<2x3xbf16>
  }
}
