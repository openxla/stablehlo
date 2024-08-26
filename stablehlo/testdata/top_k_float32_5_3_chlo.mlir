// RUN: stablehlo-opt --chlo-pre-serialization-pipeline -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-opt --chlo-pre-serialization-pipeline %s | stablehlo-translate --serialize --target=current | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt --chlo-pre-serialization-pipeline %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<5x2xf32> {jax.result_info = "[0]", mhlo.layout_mode = "default"}, tensor<5x2xi32> {jax.result_info = "[1]", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<5x3xf32>
    %1:2 = call @expected() : () -> (tensor<5x2xf32>, tensor<5x2xi32>)
    %values, %indices = chlo.top_k(%0, k = 2) : tensor<5x3xf32> -> (tensor<5x2xf32>, tensor<5x2xi32>)
    stablehlo.custom_call @check.expect_eq(%values, %1#0) {has_side_effect = true} : (tensor<5x2xf32>, tensor<5x2xf32>) -> ()
    stablehlo.custom_call @check.expect_eq(%indices, %1#1) {has_side_effect = true} : (tensor<5x2xi32>, tensor<5x2xi32>) -> ()
    return %values, %indices : tensor<5x2xf32>, tensor<5x2xi32>
  }
  func.func private @inputs() -> (tensor<5x3xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[1.87676096, -1.96814215, 0.339208782], [-1.78530681, 1.6039784, -1.19955933], [0.210248947, 2.594690e+00, -0.506774485], [2.07919931, -3.76786542, -2.9934845], [0.578959584, -0.907130658, -0.925940394]]> : tensor<5x3xf32>
    return %cst : tensor<5x3xf32>
  }
  func.func private @expected() -> (tensor<5x2xf32> {mhlo.layout_mode = "default"}, tensor<5x2xi32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[1.87676096, 0.339208782], [1.6039784, -1.19955933], [2.594690e+00, 0.210248947], [2.07919931, -2.9934845], [0.578959584, -0.907130658]]> : tensor<5x2xf32>
    %c = stablehlo.constant dense<[[0, 2], [1, 2], [1, 0], [0, 2], [0, 1]]> : tensor<5x2xi32>
    return %cst, %c : tensor<5x2xf32>, tensor<5x2xi32>
  }
}
