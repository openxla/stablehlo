// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3xcomplex<f64>> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<2x3xcomplex<f64>>
    %1 = call @expected() : () -> tensor<3xcomplex<f64>>
    %cst = stablehlo.constant dense<(1.000000e+00,0.000000e+00)> : tensor<complex<f64>>
    %2 = stablehlo.reduce(%0 init: %cst) applies stablehlo.multiply across dimensions = [0] : (tensor<2x3xcomplex<f64>>, tensor<complex<f64>>) -> tensor<3xcomplex<f64>>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<3xcomplex<f64>>, tensor<3xcomplex<f64>>) -> ()
    return %2 : tensor<3xcomplex<f64>>
  }
  func.func private @inputs() -> (tensor<2x3xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[(-3.7075252346119498,2.3747053267110676), (-2.8248724820127937,-0.43379307192698208), (-4.3787338274375687,3.667746913686682)], [(1.8555001818174246,1.886921310804297), (0.52301083883203736,-2.0290334106508694), (-1.5719747328314484,0.35950850655637528)]]> : tensor<2x3xcomplex<f64>>
    return %cst : tensor<2x3xcomplex<f64>>
  }
  func.func private @expected() -> (tensor<3xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[(-11.360195834766756,-2.5895422100587964), (-2.3576195626597731,5.5048821684041487), (5.5646727231599433,-7.3397975336459531)]> : tensor<3xcomplex<f64>>
    return %cst : tensor<3xcomplex<f64>>
  }
}
