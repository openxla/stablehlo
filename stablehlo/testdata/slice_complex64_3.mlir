// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<1xcomplex<f32>> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<3xcomplex<f32>>
    %1 = call @expected() : () -> tensor<1xcomplex<f32>>
    %2 = stablehlo.slice %0 [1:2] : (tensor<3xcomplex<f32>>) -> tensor<1xcomplex<f32>>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<1xcomplex<f32>>, tensor<1xcomplex<f32>>) -> ()
    return %2 : tensor<1xcomplex<f32>>
  }
  func.func private @inputs() -> (tensor<3xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[(1.87978518,-1.79707468), (3.20142984,-3.70751119), (0.516356587,2.9530642)]> : tensor<3xcomplex<f32>>
    return %cst : tensor<3xcomplex<f32>>
  }
  func.func private @expected() -> (tensor<1xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<(3.20142984,-3.70751119)> : tensor<1xcomplex<f32>>
    return %cst : tensor<1xcomplex<f32>>
  }
}
