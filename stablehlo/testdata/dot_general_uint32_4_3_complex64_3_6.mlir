// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x6xcomplex<f32>> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<4x3xui32>, tensor<3x6xcomplex<f32>>)
    %1 = call @expected() : () -> tensor<4x6xcomplex<f32>>
    %2 = stablehlo.convert %0#0 : (tensor<4x3xui32>) -> tensor<4x3xcomplex<f32>>
    %3 = stablehlo.convert %0#1 : tensor<3x6xcomplex<f32>>
    %4 = stablehlo.dot_general %2, %3, contracting_dims = [1] x [0] : (tensor<4x3xcomplex<f32>>, tensor<3x6xcomplex<f32>>) -> tensor<4x6xcomplex<f32>>
    stablehlo.custom_call @check.expect_close(%4, %1) {has_side_effect = true} : (tensor<4x6xcomplex<f32>>, tensor<4x6xcomplex<f32>>) -> ()
    return %4 : tensor<4x6xcomplex<f32>>
  }
  func.func private @inputs() -> (tensor<4x3xui32> {mhlo.layout_mode = "default"}, tensor<3x6xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[6, 5, 2], [8, 1, 1], [0, 1, 0], [3, 0, 1]]> : tensor<4x3xui32>
    %cst = stablehlo.constant dense<[[(-1.28161216,2.83996654), (4.59063721,3.53628182), (2.4025712,0.227394387), (-3.95283508,-4.72535372), (3.01752424,2.18881536), (-4.4802928,-2.52829719)], [(0.522764087,-2.36475348), (-1.79787827,1.78881705), (0.0115512172,-0.875928103), (2.14484739,-1.69148505), (-1.7328558,-1.31126142), (-0.684728682,-3.51046896)], [(-5.26408863,0.234941348), (0.240208596,-0.955130398), (-2.31474972,1.15676379), (-1.94263303,2.95949149), (-0.112201691,3.06944656), (-1.50385356,3.05907798)]]> : tensor<3x6xcomplex<f32>>
    return %c, %cst : tensor<4x3xui32>, tensor<3x6xcomplex<f32>>
  }
  func.func private @expected() -> (tensor<4x6xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[(-15.6040297,5.68591356), (19.0348492,28.2515163), (9.84368419,-0.701746702), (-16.8780384,-30.890564), (9.21646213,12.7154789), (-33.3131065,-26.6039734)], [(-14.9942217,20.58992), (35.1674271,29.1239414), (16.9173698,2.09999084), (-31.4204655,-36.5348244), (22.2951355,19.2687073), (-38.0309258,-20.6777687)], [(0.522764087,-2.36475348), (-1.79787827,1.78881705), (0.0115512172,-0.875928103), (2.14484739,-1.69148505), (-1.7328558,-1.31126142), (-0.684728682,-3.51046896)], [(-9.10892486,8.75484085), (14.0121202,9.65371513), (4.89296389,1.83894694), (-13.8011379,-11.2165699), (8.94037055,9.63589286), (-14.9447317,-4.5258131)]]> : tensor<4x6xcomplex<f32>>
    return %cst : tensor<4x6xcomplex<f32>>
  }
}
