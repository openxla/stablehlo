// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<2x3xcomplex<f32>> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:4 = call @inputs() : () -> (tensor<2x3xi32>, tensor<2x3xcomplex<f32>>, tensor<2x3xcomplex<f32>>, tensor<2x3xcomplex<f32>>)
    %1 = call @expected() : () -> tensor<2x3xcomplex<f32>>
    %c = stablehlo.constant dense<1> : tensor<i32>
    %2 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<2x3xi32>
    %3 = stablehlo.compare  LT, %0#0, %2,  SIGNED : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi1>
    %c_0 = stablehlo.constant dense<2> : tensor<i32>
    %4 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i32>) -> tensor<2x3xi32>
    %5 = stablehlo.compare  LT, %0#0, %4,  SIGNED : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi1>
    %6 = stablehlo.select %5, %0#2, %0#3 : tensor<2x3xi1>, tensor<2x3xcomplex<f32>>
    %7 = stablehlo.select %3, %0#1, %6 : tensor<2x3xi1>, tensor<2x3xcomplex<f32>>
    stablehlo.custom_call @check.expect_close(%7, %1) {has_side_effect = true} : (tensor<2x3xcomplex<f32>>, tensor<2x3xcomplex<f32>>) -> ()
    return %7 : tensor<2x3xcomplex<f32>>
  }
  func.func private @inputs() -> (tensor<2x3xi32> {mhlo.layout_mode = "default"}, tensor<2x3xcomplex<f32>> {mhlo.layout_mode = "default"}, tensor<2x3xcomplex<f32>> {mhlo.layout_mode = "default"}, tensor<2x3xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[2, 2, 2], [2, 1, 2]]> : tensor<2x3xi32>
    %cst = stablehlo.constant dense<[[(3.93428254,0.910755396), (-0.824884116,6.81380701), (0.620987177,-2.56805634)], [(3.92014217,-4.66776276), (-3.18807101,0.409840196), (-4.76146793,0.942774832)]]> : tensor<2x3xcomplex<f32>>
    %cst_0 = stablehlo.constant dense<[[(-3.53006744,2.75585723), (6.491930e+00,-2.32900286), (1.81959879,2.03441906)], [(0.980624616,1.46047854), (-2.48105192,1.55319786), (-2.05409145,1.94431758)]]> : tensor<2x3xcomplex<f32>>
    %cst_1 = stablehlo.constant dense<[[(-0.700041056,-2.54219437), (-3.90275693,-2.60665703), (-1.46083939,0.337038815)], [(-2.17983437,1.1594137), (-0.577557862,0.929602205), (2.11863852,-0.408816457)]]> : tensor<2x3xcomplex<f32>>
    return %c, %cst, %cst_0, %cst_1 : tensor<2x3xi32>, tensor<2x3xcomplex<f32>>, tensor<2x3xcomplex<f32>>, tensor<2x3xcomplex<f32>>
  }
  func.func private @expected() -> (tensor<2x3xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[(-0.700041056,-2.54219437), (-3.90275693,-2.60665703), (-1.46083939,0.337038815)], [(-2.17983437,1.1594137), (-2.48105192,1.55319786), (2.11863852,-0.408816457)]]> : tensor<2x3xcomplex<f32>>
    return %cst : tensor<2x3xcomplex<f32>>
  }
}
