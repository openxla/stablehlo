// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3xcomplex<f64>> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<2x3xcomplex<f64>>
    %1 = call @expected() : () -> tensor<3xcomplex<f64>>
    %cst = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<complex<f64>>
    %2 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<2x3xcomplex<f64>>, tensor<complex<f64>>) -> tensor<3xcomplex<f64>>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<3xcomplex<f64>>, tensor<3xcomplex<f64>>) -> ()
    return %2 : tensor<3xcomplex<f64>>
  }
  func.func private @inputs() -> (tensor<2x3xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[(-5.0016637729619955,-2.7226144432763526), (0.30071536304071916,-0.31702764635348196), (2.2561604067910577,-0.52579407344003848)], [(-0.38841319310607747,4.0382795230843636), (0.52697596861662555,2.4393389058990746), (0.31194372240006796,0.59519467929679115)]]> : tensor<2x3xcomplex<f64>>
    return %cst : tensor<2x3xcomplex<f64>>
  }
  func.func private @expected() -> (tensor<3xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[(-5.3900769660680732,1.315665079808011), (0.82769133165734465,2.1223112595455929), (2.5681041291911257,0.069400605856752673)]> : tensor<3xcomplex<f64>>
    return %cst : tensor<3xcomplex<f64>>
  }
}
