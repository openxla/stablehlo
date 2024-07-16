// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<2x3xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:3 = call @inputs() : () -> (tensor<2x3xf32>, tensor<2x3xf32>, tensor<2x3xf32>)
    %1 = call @expected() : () -> tensor<2x3xf32>
    %2 = stablehlo.clamp %0#0, %0#1, %0#2 : tensor<2x3xf32>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<2x3xf32>, tensor<2x3xf32>) -> ()
    return %2 : tensor<2x3xf32>
  }
  func.func private @inputs() -> (tensor<2x3xf32> {mhlo.layout_mode = "default"}, tensor<2x3xf32> {mhlo.layout_mode = "default"}, tensor<2x3xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-0.890106558, -2.546803, 3.41218138], [-2.42367935, 0.556205094, -4.37419939]]> : tensor<2x3xf32>
    %cst_0 = stablehlo.constant dense<[[-0.682396054, -5.85568476, 5.77086878], [-4.9932375, -2.85505986, -4.2979579]]> : tensor<2x3xf32>
    %cst_1 = stablehlo.constant dense<[[-2.02943087, 1.23262703, -2.46976185], [4.53440619, 2.10693359, -3.85020328]]> : tensor<2x3xf32>
    return %cst, %cst_0, %cst_1 : tensor<2x3xf32>, tensor<2x3xf32>, tensor<2x3xf32>
  }
  func.func private @expected() -> (tensor<2x3xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-2.02943087, -2.546803, -2.46976185], [-2.42367935, 0.556205094, -4.2979579]]> : tensor<2x3xf32>
    return %cst : tensor<2x3xf32>
  }
}
