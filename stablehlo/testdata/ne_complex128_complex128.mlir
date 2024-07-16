// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<i1> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<complex<f64>>, tensor<complex<f64>>)
    %1 = call @expected() : () -> tensor<i1>
    %2 = stablehlo.compare  NE, %0#0, %0#1,  FLOAT : (tensor<complex<f64>>, tensor<complex<f64>>) -> tensor<i1>
    stablehlo.custom_call @check.expect_eq(%2, %1) {has_side_effect = true} : (tensor<i1>, tensor<i1>) -> ()
    return %2 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<complex<f64>> {mhlo.layout_mode = "default"}, tensor<complex<f64>> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<(-3.5966994797917264,1.0583004032167105)> : tensor<complex<f64>>
    %cst_0 = stablehlo.constant dense<(3.7422441882481152,1.8148800236006304)> : tensor<complex<f64>>
    return %cst, %cst_0 : tensor<complex<f64>>, tensor<complex<f64>>
  }
  func.func private @expected() -> (tensor<i1> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<true> : tensor<i1>
    return %c : tensor<i1>
  }
}
