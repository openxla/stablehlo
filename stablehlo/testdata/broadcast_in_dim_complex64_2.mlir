// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<2xcomplex<f32>> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<2xcomplex<f32>>
    %1 = call @expected() : () -> tensor<2xcomplex<f32>>
    stablehlo.custom_call @check.expect_close(%0, %1) {has_side_effect = true} : (tensor<2xcomplex<f32>>, tensor<2xcomplex<f32>>) -> ()
    return %0 : tensor<2xcomplex<f32>>
  }
  func.func private @inputs() -> (tensor<2xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[(-1.8468895,0.316099197), (4.64661932,-2.96641493)]> : tensor<2xcomplex<f32>>
    return %cst : tensor<2xcomplex<f32>>
  }
  func.func private @expected() -> (tensor<2xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[(-1.8468895,0.316099197), (4.64661932,-2.96641493)]> : tensor<2xcomplex<f32>>
    return %cst : tensor<2xcomplex<f32>>
  }
}
