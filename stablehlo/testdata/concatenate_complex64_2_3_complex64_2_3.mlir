// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x3xcomplex<f32>> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<2x3xcomplex<f32>>, tensor<2x3xcomplex<f32>>)
    %1 = call @expected() : () -> tensor<4x3xcomplex<f32>>
    %2 = stablehlo.concatenate %0#0, %0#1, dim = 0 : (tensor<2x3xcomplex<f32>>, tensor<2x3xcomplex<f32>>) -> tensor<4x3xcomplex<f32>>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<4x3xcomplex<f32>>, tensor<4x3xcomplex<f32>>) -> ()
    return %2 : tensor<4x3xcomplex<f32>>
  }
  func.func private @inputs() -> (tensor<2x3xcomplex<f32>> {mhlo.layout_mode = "default"}, tensor<2x3xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[(1.07382131,-2.67878866), (0.427812099,4.57176638), (5.087150e+00,-0.220292255)], [(1.57771051,2.09996724), (2.37849593,-3.05391335), (0.732661545,-5.76225805)]]> : tensor<2x3xcomplex<f32>>
    %cst_0 = stablehlo.constant dense<[[(-1.83474767,2.32461143), (-0.989446878,-0.815832734), (5.39325857,2.16857696)], [(2.27715683,-0.268771797), (0.337234288,3.84130883), (-7.95383501,-0.700022578)]]> : tensor<2x3xcomplex<f32>>
    return %cst, %cst_0 : tensor<2x3xcomplex<f32>>, tensor<2x3xcomplex<f32>>
  }
  func.func private @expected() -> (tensor<4x3xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[(1.07382131,-2.67878866), (0.427812099,4.57176638), (5.087150e+00,-0.220292255)], [(1.57771051,2.09996724), (2.37849593,-3.05391335), (0.732661545,-5.76225805)], [(-1.83474767,2.32461143), (-0.989446878,-0.815832734), (5.39325857,2.16857696)], [(2.27715683,-0.268771797), (0.337234288,3.84130883), (-7.95383501,-0.700022578)]]> : tensor<4x3xcomplex<f32>>
    return %cst : tensor<4x3xcomplex<f32>>
  }
}
