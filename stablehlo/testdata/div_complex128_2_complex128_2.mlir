// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<2xcomplex<f64>> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<2xcomplex<f64>>, tensor<2xcomplex<f64>>)
    %1 = call @expected() : () -> tensor<2xcomplex<f64>>
    %2 = stablehlo.divide %0#0, %0#1 : tensor<2xcomplex<f64>>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<2xcomplex<f64>>, tensor<2xcomplex<f64>>) -> ()
    return %2 : tensor<2xcomplex<f64>>
  }
  func.func private @inputs() -> (tensor<2xcomplex<f64>> {mhlo.layout_mode = "default"}, tensor<2xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[(-0.4142508652598253,2.9708064697134127), (-0.086255762317640316,-2.1008513159963607)]> : tensor<2xcomplex<f64>>
    %cst_0 = stablehlo.constant dense<[(-2.3701009044089654,3.734864852692362), (1.6009430068468937,-0.20405409383852913)]> : tensor<2xcomplex<f64>>
    return %cst, %cst_0 : tensor<2xcomplex<f64>>, tensor<2xcomplex<f64>>
  }
  func.func private @expected() -> (tensor<2xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[(0.61724473678798175,-0.28078163066250666), (0.11156816366298221,-1.2980383227706118)]> : tensor<2xcomplex<f64>>
    return %cst : tensor<2xcomplex<f64>>
  }
}
