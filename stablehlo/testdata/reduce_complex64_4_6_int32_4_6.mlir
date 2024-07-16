// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<6xcomplex<f32>> {jax.result_info = "[0]", mhlo.layout_mode = "default"}, tensor<6xi32> {jax.result_info = "[1]", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<4x6xcomplex<f32>>, tensor<4x6xi32>)
    %1:2 = call @expected() : () -> (tensor<6xcomplex<f32>>, tensor<6xi32>)
    %cst = stablehlo.constant dense<(3.000000e+00,0.000000e+00)> : tensor<complex<f32>>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %2:2 = stablehlo.reduce(%0#0 init: %cst), (%0#1 init: %c) across dimensions = [0] : (tensor<4x6xcomplex<f32>>, tensor<4x6xi32>, tensor<complex<f32>>, tensor<i32>) -> (tensor<6xcomplex<f32>>, tensor<6xi32>)
     reducer(%arg0: tensor<complex<f32>>, %arg2: tensor<complex<f32>>) (%arg1: tensor<i32>, %arg3: tensor<i32>)  {
      %3 = stablehlo.real %arg0 : (tensor<complex<f32>>) -> tensor<f32>
      %4 = stablehlo.real %arg2 : (tensor<complex<f32>>) -> tensor<f32>
      %5 = stablehlo.compare  EQ, %3, %4,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %6 = stablehlo.compare  GT, %3, %4,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %7 = stablehlo.imag %arg0 : (tensor<complex<f32>>) -> tensor<f32>
      %8 = stablehlo.imag %arg2 : (tensor<complex<f32>>) -> tensor<f32>
      %9 = stablehlo.compare  GT, %7, %8,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %10 = stablehlo.select %5, %9, %6 : tensor<i1>, tensor<i1>
      %11 = stablehlo.select %10, %arg0, %arg2 : tensor<i1>, tensor<complex<f32>>
      %12 = stablehlo.minimum %arg1, %arg3 : tensor<i32>
      stablehlo.return %11, %12 : tensor<complex<f32>>, tensor<i32>
    }
    stablehlo.custom_call @check.expect_close(%2#0, %1#0) {has_side_effect = true} : (tensor<6xcomplex<f32>>, tensor<6xcomplex<f32>>) -> ()
    stablehlo.custom_call @check.expect_eq(%2#1, %1#1) {has_side_effect = true} : (tensor<6xi32>, tensor<6xi32>) -> ()
    return %2#0, %2#1 : tensor<6xcomplex<f32>>, tensor<6xi32>
  }
  func.func private @inputs() -> (tensor<4x6xcomplex<f32>> {mhlo.layout_mode = "default"}, tensor<4x6xi32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[(-0.381778359,-1.21680665), (-1.29195714,-5.36601257), (-3.14889767E-4,-1.04874849), (1.17392564,2.8273983), (0.539779663,1.59845591), (3.501680e+00,-1.58961439)], [(-8.656080e-02,-3.4427917), (1.70420969,1.87724805), (-2.88715482,1.74130571), (-1.96685147,2.44956136), (-0.395261019,0.492172718), (-2.60446668,-0.676473498)], [(-0.582556963,-7.35778141), (2.14501739,-2.86162686), (4.00450563,5.36438274), (0.776385069,-2.65484333), (-1.4155364,4.91823721), (-5.45412445,0.0295851938)], [(0.297839791,2.33191633), (4.9744482,-5.89148521), (-1.80037713,1.70301712), (-1.21536481,-5.40402555), (-0.411337882,1.20069647), (0.307315618,-2.6319325)]]> : tensor<4x6xcomplex<f32>>
    %c = stablehlo.constant dense<[[4, -1, 0, 2, -1, 2], [0, -1, -4, 5, 0, 1], [-5, -2, -1, -2, 5, -4], [1, 0, -4, 0, 0, -1]]> : tensor<4x6xi32>
    return %cst, %c : tensor<4x6xcomplex<f32>>, tensor<4x6xi32>
  }
  func.func private @expected() -> (tensor<6xcomplex<f32>> {mhlo.layout_mode = "default"}, tensor<6xi32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[(3.000000e+00,0.000000e+00), (4.9744482,-5.89148521), (4.00450563,5.36438274), (3.000000e+00,0.000000e+00), (3.000000e+00,0.000000e+00), (3.501680e+00,-1.58961439)]> : tensor<6xcomplex<f32>>
    %c = stablehlo.constant dense<[-5, -2, -4, -2, -1, -4]> : tensor<6xi32>
    return %cst, %c : tensor<6xcomplex<f32>>, tensor<6xi32>
  }
}
