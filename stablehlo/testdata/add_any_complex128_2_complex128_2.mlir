// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<2xcomplex<f64>> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<2xcomplex<f64>>, tensor<2xcomplex<f64>>)
    %1 = call @expected() : () -> tensor<2xcomplex<f64>>
    %2 = stablehlo.add %0#0, %0#1 : tensor<2xcomplex<f64>>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<2xcomplex<f64>>, tensor<2xcomplex<f64>>) -> ()
    return %2 : tensor<2xcomplex<f64>>
  }
  func.func private @inputs() -> (tensor<2xcomplex<f64>> {mhlo.layout_mode = "default"}, tensor<2xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[(0.42742970398722779,0.19525882468777409), (-1.5782763149646164,-4.0015414451742641)]> : tensor<2xcomplex<f64>>
    %cst_0 = stablehlo.constant dense<[(-0.11290775549527821,-3.1382975860447448), (2.7883747974901301,-1.8044923462614515)]> : tensor<2xcomplex<f64>>
    return %cst, %cst_0 : tensor<2xcomplex<f64>>, tensor<2xcomplex<f64>>
  }
  func.func private @expected() -> (tensor<2xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[(0.31452194849194959,-2.9430387613569708), (1.2100984825255137,-5.8060337914357154)]> : tensor<2xcomplex<f64>>
    return %cst : tensor<2xcomplex<f64>>
  }
}
