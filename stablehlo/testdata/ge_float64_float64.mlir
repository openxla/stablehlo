// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<i1> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<f64>, tensor<f64>)
    %1 = call @expected() : () -> tensor<i1>
    %2 = stablehlo.compare  GE, %0#0, %0#1,  FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<i1>, tensor<i1>) -> ()
    return %2 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<f64> {mhlo.layout_mode = "default"}, tensor<f64> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<-1.2435148516044192> : tensor<f64>
    %cst_0 = stablehlo.constant dense<-1.3191157376300089> : tensor<f64>
    return %cst, %cst_0 : tensor<f64>, tensor<f64>
  }
  func.func private @expected() -> (tensor<i1> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<true> : tensor<i1>
    return %c : tensor<i1>
  }
}
