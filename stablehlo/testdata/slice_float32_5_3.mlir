// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<2x2xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<5x3xf32>
    %1 = call @expected() : () -> tensor<2x2xf32>
    %2 = stablehlo.slice %0 [1:5:2, 1:3] : (tensor<5x3xf32>) -> tensor<2x2xf32>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<2x2xf32>, tensor<2x2xf32>) -> ()
    return %2 : tensor<2x2xf32>
  }
  func.func private @inputs() -> (tensor<5x3xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[0.551178575, 2.42043257, 6.3848033], [-1.56378138, -0.768643915, -3.33495092], [7.55508327, -0.759385645, -0.365653396], [-2.15069771, -1.417720e+00, -3.05872202], [-3.96960521, 0.531166673, -2.26626372]]> : tensor<5x3xf32>
    return %cst : tensor<5x3xf32>
  }
  func.func private @expected() -> (tensor<2x2xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-0.768643915, -3.33495092], [-1.417720e+00, -3.05872202]]> : tensor<2x2xf32>
    return %cst : tensor<2x2xf32>
  }
}
