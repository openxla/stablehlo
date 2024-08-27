// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x6xcomplex<f32>> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<4x3xi32>, tensor<3x6xcomplex<f32>>)
    %1 = call @expected() : () -> tensor<4x6xcomplex<f32>>
    %2 = stablehlo.convert %0#0 : (tensor<4x3xi32>) -> tensor<4x3xcomplex<f32>>
    %3 = stablehlo.convert %0#1 : tensor<3x6xcomplex<f32>>
    %4 = stablehlo.dot_general %2, %3, contracting_dims = [1] x [0] : (tensor<4x3xcomplex<f32>>, tensor<3x6xcomplex<f32>>) -> tensor<4x6xcomplex<f32>>
    stablehlo.custom_call @check.expect_close(%4, %1) {has_side_effect = true} : (tensor<4x6xcomplex<f32>>, tensor<4x6xcomplex<f32>>) -> ()
    return %4 : tensor<4x6xcomplex<f32>>
  }
  func.func private @inputs() -> (tensor<4x3xi32> {mhlo.layout_mode = "default"}, tensor<3x6xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[4, 0, 2], [1, -2, 0], [1, 0, -1], [-1, 0, 0]]> : tensor<4x3xi32>
    %cst = stablehlo.constant dense<[[(7.92069626,-6.47652483), (-1.3700304,2.60645723), (-2.65478921,0.21623984), (-0.563305318,3.27899218), (-0.233431309,3.66033626), (2.39221168,-4.49195242)], [(5.75185347,1.4362042), (-5.04380846,2.45227432), (0.472258747,-2.00312448), (0.310080528,3.15785241), (1.77643442,4.82490396), (-2.22111845,1.1883359)], [(1.89532757,-0.829474806), (0.437185764,-4.094920e+00), (0.919478356,5.37613535), (1.79118562,-4.25449276), (-1.19618368,1.5587045), (-3.04641867,-5.68628263)]]> : tensor<3x6xcomplex<f32>>
    return %c, %cst : tensor<4x3xi32>, tensor<3x6xcomplex<f32>>
  }
  func.func private @expected() -> (tensor<4x6xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[(35.4734421,-27.5650482), (-4.605750e+00,2.23598862), (-8.780200e+00,11.6172304), (1.329150e+00,4.60698318), (-3.32609272,17.7587547), (3.47600937,-29.340374)], [(-3.58301067,-9.34893321), (8.71758652,-2.29809141), (-3.59930658,4.22248888), (-1.18346643,-3.03671265), (-3.78630018,-5.98947144), (6.83444881,-6.86862421)], [(6.02536869,-5.647050e+00), (-1.80721617,6.70137739), (-3.57426763,-5.15989542), (-2.354491,7.53348494), (0.962752342,2.10163164), (5.438630e+00,1.19433022)], [(-7.92069626,6.47652483), (1.3700304,-2.60645723), (2.65478921,-0.21623984), (0.563305318,-3.27899218), (0.233431309,-3.66033626), (-2.39221168,4.49195242)]]> : tensor<4x6xcomplex<f32>>
    return %cst : tensor<4x6xcomplex<f32>>
  }
}
