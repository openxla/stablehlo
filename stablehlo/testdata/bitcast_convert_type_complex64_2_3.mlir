// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<2x3xcomplex<f32>> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<2x3xcomplex<f32>>
    %1 = call @expected() : () -> tensor<2x3xcomplex<f32>>
    %2 = stablehlo.bitcast_convert %0 : (tensor<2x3xcomplex<f32>>) -> tensor<2x3xcomplex<f32>>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<2x3xcomplex<f32>>, tensor<2x3xcomplex<f32>>) -> ()
    return %2 : tensor<2x3xcomplex<f32>>
  }
  func.func private @inputs() -> (tensor<2x3xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[(-2.9432714,-0.240944356), (-1.87811852,2.883020e+00), (2.98283267,-1.20440769)], [(-3.73439598,-0.442267925), (-4.78169394,1.74811637), (-1.23555768,-4.14041948)]]> : tensor<2x3xcomplex<f32>>
    return %cst : tensor<2x3xcomplex<f32>>
  }
  func.func private @expected() -> (tensor<2x3xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[(-2.9432714,-0.240944356), (-1.87811852,2.883020e+00), (2.98283267,-1.20440769)], [(-3.73439598,-0.442267925), (-4.78169394,1.74811637), (-1.23555768,-4.14041948)]]> : tensor<2x3xcomplex<f32>>
    return %cst : tensor<2x3xcomplex<f32>>
  }
}
