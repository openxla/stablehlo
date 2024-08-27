// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3xf64> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<2x3xf64>
    %1 = call @expected() : () -> tensor<3xf64>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %2 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<2x3xf64>, tensor<f64>) -> tensor<3xf64>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<3xf64>, tensor<3xf64>) -> ()
    return %2 : tensor<3xf64>
  }
  func.func private @inputs() -> (tensor<2x3xf64> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[0.70408858161780585, 0.31273915401217989, 5.168833273313818], [1.8067394690884413, 1.1213150007469181, 2.7314344185474]]> : tensor<2x3xf64>
    return %cst : tensor<2x3xf64>
  }
  func.func private @expected() -> (tensor<3xf64> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[2.5108280507062473, 1.4340541547590981, 7.9002676918612185]> : tensor<3xf64>
    return %cst : tensor<3xf64>
  }
}
