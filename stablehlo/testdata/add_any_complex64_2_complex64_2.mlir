// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<2xcomplex<f32>> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<2xcomplex<f32>>, tensor<2xcomplex<f32>>)
    %1 = call @expected() : () -> tensor<2xcomplex<f32>>
    %2 = stablehlo.add %0#0, %0#1 : tensor<2xcomplex<f32>>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<2xcomplex<f32>>, tensor<2xcomplex<f32>>) -> ()
    return %2 : tensor<2xcomplex<f32>>
  }
  func.func private @inputs() -> (tensor<2xcomplex<f32>> {mhlo.layout_mode = "default"}, tensor<2xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[(-0.992783546,-0.340118021), (-2.07474065,-1.85044086)]> : tensor<2xcomplex<f32>>
    %cst_0 = stablehlo.constant dense<[(-1.89556551,0.111282185), (-1.76712668,-0.406201214)]> : tensor<2xcomplex<f32>>
    return %cst, %cst_0 : tensor<2xcomplex<f32>>, tensor<2xcomplex<f32>>
  }
  func.func private @expected() -> (tensor<2xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[(-2.88834906,-0.228835836), (-3.84186745,-2.2566421)]> : tensor<2xcomplex<f32>>
    return %cst : tensor<2xcomplex<f32>>
  }
}
