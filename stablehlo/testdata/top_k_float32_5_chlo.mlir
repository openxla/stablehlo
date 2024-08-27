// RUN-DISABLED(#2497): stablehlo-opt --chlo-pre-serialization-pipeline -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-opt --chlo-pre-serialization-pipeline %s | stablehlo-translate --serialize --target=current | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt --chlo-pre-serialization-pipeline %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<5xf32> {jax.result_info = "[0]", mhlo.layout_mode = "default"}, tensor<5xi32> {jax.result_info = "[1]", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<5xf32>
    %1:2 = call @expected() : () -> (tensor<5xf32>, tensor<5xi32>)
    %values, %indices = chlo.top_k(%0, k = 5) : tensor<5xf32> -> (tensor<5xf32>, tensor<5xi32>)
    stablehlo.custom_call @check.expect_eq(%values, %1#0) {has_side_effect = true} : (tensor<5xf32>, tensor<5xf32>) -> ()
    stablehlo.custom_call @check.expect_eq(%indices, %1#1) {has_side_effect = true} : (tensor<5xi32>, tensor<5xi32>) -> ()
    return %values, %indices : tensor<5xf32>, tensor<5xi32>
  }
  func.func private @inputs() -> (tensor<5xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[0x7F800000, 0x7FC00000, 0xFFC00000, 0xFF800000, 3.000000e+00]> : tensor<5xf32>
    return %cst : tensor<5xf32>
  }
  func.func private @expected() -> (tensor<5xf32> {mhlo.layout_mode = "default"}, tensor<5xi32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[0x7FC00000, 0x7F800000, 3.000000e+00, 0xFF800000, 0xFFC00000]> : tensor<5xf32>
    %c = stablehlo.constant dense<[1, 0, 4, 3, 2]> : tensor<5xi32>
    return %cst, %c : tensor<5xf32>, tensor<5xi32>
  }
}
