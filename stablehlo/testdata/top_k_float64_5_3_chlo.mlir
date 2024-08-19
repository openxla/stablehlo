// RUN: stablehlo-opt --chlo-pre-serialization-pipeline -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-opt --chlo-pre-serialization-pipeline %s | stablehlo-translate --serialize --target=current | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt --chlo-pre-serialization-pipeline %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<5x2xf64> {jax.result_info = "[0]", mhlo.layout_mode = "default"}, tensor<5x2xi32> {jax.result_info = "[1]", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<5x3xf64>
    %1:2 = call @expected() : () -> (tensor<5x2xf64>, tensor<5x2xi32>)
    %values, %indices = chlo.top_k(%0, k = 2) : tensor<5x3xf64> -> (tensor<5x2xf64>, tensor<5x2xi32>)
    stablehlo.custom_call @check.expect_eq(%values, %1#0) {has_side_effect = true} : (tensor<5x2xf64>, tensor<5x2xf64>) -> ()
    stablehlo.custom_call @check.expect_eq(%indices, %1#1) {has_side_effect = true} : (tensor<5x2xi32>, tensor<5x2xi32>) -> ()
    return %values, %indices : tensor<5x2xf64>, tensor<5x2xi32>
  }
  func.func private @inputs() -> (tensor<5x3xf64> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[1.0343520378830138, -0.72005861577891961, 0.8116893130582965], [1.607892529832593, 0.98557547257388722, 2.4222800833698059], [-1.9816222607811027, -1.7198411085428242, -6.5269585694018968], [-2.0504014852139987, 0.91583800697560958, 3.4342856142046685], [-3.0949193319215005, -3.6759015659403387, 6.5702934608478234]]> : tensor<5x3xf64>
    return %cst : tensor<5x3xf64>
  }
  func.func private @expected() -> (tensor<5x2xf64> {mhlo.layout_mode = "default"}, tensor<5x2xi32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[1.0343520378830138, 0.8116893130582965], [2.4222800833698059, 1.607892529832593], [-1.7198411085428242, -1.9816222607811027], [3.4342856142046685, 0.91583800697560958], [6.5702934608478234, -3.0949193319215005]]> : tensor<5x2xf64>
    %c = stablehlo.constant dense<[[0, 2], [2, 0], [1, 0], [2, 1], [2, 0]]> : tensor<5x2xi32>
    return %cst, %c : tensor<5x2xf64>, tensor<5x2xi32>
  }
}
