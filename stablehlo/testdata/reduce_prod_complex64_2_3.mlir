// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3xcomplex<f32>> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<2x3xcomplex<f32>>
    %1 = call @expected() : () -> tensor<3xcomplex<f32>>
    %cst = stablehlo.constant dense<(1.000000e+00,0.000000e+00)> : tensor<complex<f32>>
    %2 = stablehlo.reduce(%0 init: %cst) applies stablehlo.multiply across dimensions = [0] : (tensor<2x3xcomplex<f32>>, tensor<complex<f32>>) -> tensor<3xcomplex<f32>>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<3xcomplex<f32>>, tensor<3xcomplex<f32>>) -> ()
    return %2 : tensor<3xcomplex<f32>>
  }
  func.func private @inputs() -> (tensor<2x3xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[(-1.21528089,-2.54824376), (2.66414642,-0.670537173), (-1.84409964,-2.86288095)], [(-1.61518693,2.00412917), (0.720621347,-0.149525791), (1.0502249,4.87241888)]]> : tensor<2x3xcomplex<f32>>
    return %cst : tensor<2x3xcomplex<f32>>
  }
  func.func private @expected() -> (tensor<3xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[(7.06991577,1.68031025), (1.81957817,-8.815620e-01), (12.0124359,-11.9918947)]> : tensor<3xcomplex<f32>>
    return %cst : tensor<3xcomplex<f32>>
  }
}
