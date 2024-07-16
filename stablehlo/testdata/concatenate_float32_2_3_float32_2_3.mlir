// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<2x6xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<2x3xf32>, tensor<2x3xf32>)
    %1 = call @expected() : () -> tensor<2x6xf32>
    %2 = stablehlo.concatenate %0#0, %0#1, dim = 1 : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x6xf32>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<2x6xf32>, tensor<2x6xf32>) -> ()
    return %2 : tensor<2x6xf32>
  }
  func.func private @inputs() -> (tensor<2x3xf32> {mhlo.layout_mode = "default"}, tensor<2x3xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-2.29547954, 2.90465164, -6.267250e+00], [1.80744612, -0.891328752, -5.21437025]]> : tensor<2x3xf32>
    %cst_0 = stablehlo.constant dense<[[2.62669659, 0.361944199, -6.23034668], [5.21641445, 0.966221809, 3.38860106]]> : tensor<2x3xf32>
    return %cst, %cst_0 : tensor<2x3xf32>, tensor<2x3xf32>
  }
  func.func private @expected() -> (tensor<2x6xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-2.29547954, 2.90465164, -6.267250e+00, 2.62669659, 0.361944199, -6.23034668], [1.80744612, -0.891328752, -5.21437025, 5.21641445, 0.966221809, 3.38860106]]> : tensor<2x6xf32>
    return %cst : tensor<2x6xf32>
  }
}
