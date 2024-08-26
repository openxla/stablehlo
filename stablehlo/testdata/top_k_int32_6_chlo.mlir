// RUN: stablehlo-opt --chlo-pre-serialization-pipeline -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-opt --chlo-pre-serialization-pipeline %s | stablehlo-translate --serialize --target=current | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt --chlo-pre-serialization-pipeline %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3xi32> {jax.result_info = "[0]", mhlo.layout_mode = "default"}, tensor<3xi32> {jax.result_info = "[1]", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<6xi32>
    %1:2 = call @expected() : () -> (tensor<3xi32>, tensor<3xi32>)
    %values, %indices = chlo.top_k(%0, k = 3) : tensor<6xi32> -> (tensor<3xi32>, tensor<3xi32>)
    stablehlo.custom_call @check.expect_eq(%values, %1#0) {has_side_effect = true} : (tensor<3xi32>, tensor<3xi32>) -> ()
    stablehlo.custom_call @check.expect_eq(%indices, %1#1) {has_side_effect = true} : (tensor<3xi32>, tensor<3xi32>) -> ()
    return %values, %indices : tensor<3xi32>, tensor<3xi32>
  }
  func.func private @inputs() -> (tensor<6xi32> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[5, 7, 5, 8, 8, 5]> : tensor<6xi32>
    return %c : tensor<6xi32>
  }
  func.func private @expected() -> (tensor<3xi32> {mhlo.layout_mode = "default"}, tensor<3xi32> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[8, 8, 7]> : tensor<3xi32>
    %c_0 = stablehlo.constant dense<[3, 4, 1]> : tensor<3xi32>
    return %c, %c_0 : tensor<3xi32>, tensor<3xi32>
  }
}
