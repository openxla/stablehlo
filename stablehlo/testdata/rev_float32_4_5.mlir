// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x5xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<4x5xf32>
    %1 = call @expected() : () -> tensor<4x5xf32>
    %2 = stablehlo.reverse %0, dims = [0] : tensor<4x5xf32>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<4x5xf32>, tensor<4x5xf32>) -> ()
    return %2 : tensor<4x5xf32>
  }
  func.func private @inputs() -> (tensor<4x5xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[0.953698933, 0.58084029, 2.857903, 3.98109102, 2.28901577], [-0.449118882, 0.0547672883, -2.05886245, 2.94395828, -3.02962065], [1.27594137, -2.0539515, -0.698254168, 2.72901058, -4.18458509], [-4.04998112, 2.12366033, -5.84264135, -1.21474063, 2.35050321]]> : tensor<4x5xf32>
    return %cst : tensor<4x5xf32>
  }
  func.func private @expected() -> (tensor<4x5xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-4.04998112, 2.12366033, -5.84264135, -1.21474063, 2.35050321], [1.27594137, -2.0539515, -0.698254168, 2.72901058, -4.18458509], [-0.449118882, 0.0547672883, -2.05886245, 2.94395828, -3.02962065], [0.953698933, 0.58084029, 2.857903, 3.98109102, 2.28901577]]> : tensor<4x5xf32>
    return %cst : tensor<4x5xf32>
  }
}
