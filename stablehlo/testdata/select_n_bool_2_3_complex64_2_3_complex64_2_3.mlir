// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<2x3xcomplex<f32>> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:3 = call @inputs() : () -> (tensor<2x3xi1>, tensor<2x3xcomplex<f32>>, tensor<2x3xcomplex<f32>>)
    %1 = call @expected() : () -> tensor<2x3xcomplex<f32>>
    %2 = stablehlo.select %0#0, %0#2, %0#1 : tensor<2x3xi1>, tensor<2x3xcomplex<f32>>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<2x3xcomplex<f32>>, tensor<2x3xcomplex<f32>>) -> ()
    return %2 : tensor<2x3xcomplex<f32>>
  }
  func.func private @inputs() -> (tensor<2x3xi1> {mhlo.layout_mode = "default"}, tensor<2x3xcomplex<f32>> {mhlo.layout_mode = "default"}, tensor<2x3xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<true> : tensor<2x3xi1>
    %cst = stablehlo.constant dense<[[(2.43387675,4.23526859), (1.10274744,2.2652235), (-4.38410425,0.678857446)], [(0.126821086,-3.64852381), (-1.20231128,3.67738676), (3.78677344,4.46107054)]]> : tensor<2x3xcomplex<f32>>
    %cst_0 = stablehlo.constant dense<[[(3.70096087,0.259771079), (-2.55815077,0.15060997), (1.94375587,1.97517061)], [(-6.01416063,-2.1360147), (2.66556025,3.83397913), (-2.5734024,-1.7117784)]]> : tensor<2x3xcomplex<f32>>
    return %c, %cst, %cst_0 : tensor<2x3xi1>, tensor<2x3xcomplex<f32>>, tensor<2x3xcomplex<f32>>
  }
  func.func private @expected() -> (tensor<2x3xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[(3.70096087,0.259771079), (-2.55815077,0.15060997), (1.94375587,1.97517061)], [(-6.01416063,-2.1360147), (2.66556025,3.83397913), (-2.5734024,-1.7117784)]]> : tensor<2x3xcomplex<f32>>
    return %cst : tensor<2x3xcomplex<f32>>
  }
}
