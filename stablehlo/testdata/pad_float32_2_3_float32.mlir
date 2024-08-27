// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<2x0xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<2x3xf32>, tensor<f32>)
    %1 = call @expected() : () -> tensor<2x0xf32>
    %2 = stablehlo.pad %0#0, %0#1, low = [0, -2], high = [0, -3], interior = [0, 1] : (tensor<2x3xf32>, tensor<f32>) -> tensor<2x0xf32>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<2x0xf32>, tensor<2x0xf32>) -> ()
    return %2 : tensor<2x0xf32>
  }
  func.func private @inputs() -> (tensor<2x3xf32> {mhlo.layout_mode = "default"}, tensor<f32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[2.38316075E-4, -1.74506466E-4, 0.00169209205], [3.11927375E-4, 0.00166800153, 6.94549758E-4]]> : tensor<2x3xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    return %cst, %cst_0 : tensor<2x3xf32>, tensor<f32>
  }
  func.func private @expected() -> (tensor<2x0xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<> : tensor<2x0xf32>
    return %cst : tensor<2x0xf32>
  }
}
