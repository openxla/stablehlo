// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<2x3xf64> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:3 = call @inputs() : () -> (tensor<f64>, tensor<2x3xf64>, tensor<f64>)
    %1 = call @expected() : () -> tensor<2x3xf64>
    %2 = stablehlo.broadcast_in_dim %0#0, dims = [] : (tensor<f64>) -> tensor<2x3xf64>
    %3 = stablehlo.broadcast_in_dim %0#2, dims = [] : (tensor<f64>) -> tensor<2x3xf64>
    %4 = stablehlo.clamp %2, %0#1, %3 : tensor<2x3xf64>
    stablehlo.custom_call @check.expect_close(%4, %1) {has_side_effect = true} : (tensor<2x3xf64>, tensor<2x3xf64>) -> ()
    return %4 : tensor<2x3xf64>
  }
  func.func private @inputs() -> (tensor<f64> {mhlo.layout_mode = "default"}, tensor<2x3xf64> {mhlo.layout_mode = "default"}, tensor<f64> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[1.3647748627855825, 0.55441559336763024, -2.355556198180957], [2.2665459589395178, 0.58817989765507428, 3.7514669634211639]]> : tensor<2x3xf64>
    %cst_0 = stablehlo.constant dense<2.6354240164456249> : tensor<f64>
    %cst_1 = stablehlo.constant dense<4.0043262680460456> : tensor<f64>
    return %cst_0, %cst, %cst_1 : tensor<f64>, tensor<2x3xf64>, tensor<f64>
  }
  func.func private @expected() -> (tensor<2x3xf64> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[2.6354240164456249, 2.6354240164456249, 2.6354240164456249], [2.6354240164456249, 2.6354240164456249, 3.7514669634211639]]> : tensor<2x3xf64>
    return %cst : tensor<2x3xf64>
  }
}
