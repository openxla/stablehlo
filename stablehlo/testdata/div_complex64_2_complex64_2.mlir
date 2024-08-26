// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<2xcomplex<f32>> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<2xcomplex<f32>>, tensor<2xcomplex<f32>>)
    %1 = call @expected() : () -> tensor<2xcomplex<f32>>
    %2 = stablehlo.divide %0#0, %0#1 : tensor<2xcomplex<f32>>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<2xcomplex<f32>>, tensor<2xcomplex<f32>>) -> ()
    return %2 : tensor<2xcomplex<f32>>
  }
  func.func private @inputs() -> (tensor<2xcomplex<f32>> {mhlo.layout_mode = "default"}, tensor<2xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[(4.82876539,1.14962423), (0.816298723,-3.73786092)]> : tensor<2xcomplex<f32>>
    %cst_0 = stablehlo.constant dense<[(-6.938800e-01,-4.50840855), (-1.38481331,3.32134724)]> : tensor<2xcomplex<f32>>
    return %cst, %cst_0 : tensor<2xcomplex<f32>>, tensor<2xcomplex<f32>>
  }
  func.func private @expected() -> (tensor<2xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[(-0.410124958,1.00793612), (-1.04603434,0.190363556)]> : tensor<2xcomplex<f32>>
    return %cst : tensor<2xcomplex<f32>>
  }
}
