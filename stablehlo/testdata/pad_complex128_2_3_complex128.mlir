// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<2x1xcomplex<f64>> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<2x3xcomplex<f64>>, tensor<complex<f64>>)
    %1 = call @expected() : () -> tensor<2x1xcomplex<f64>>
    %2 = stablehlo.pad %0#0, %0#1, low = [0, -1], high = [0, -1], interior = [0, 0] : (tensor<2x3xcomplex<f64>>, tensor<complex<f64>>) -> tensor<2x1xcomplex<f64>>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<2x1xcomplex<f64>>, tensor<2x1xcomplex<f64>>) -> ()
    return %2 : tensor<2x1xcomplex<f64>>
  }
  func.func private @inputs() -> (tensor<2x3xcomplex<f64>> {mhlo.layout_mode = "default"}, tensor<complex<f64>> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[(3.0371844027451186E-4,8.634863150028557E-5), (-0.0010977968750361337,-0.0020786308515255291), (-4.906680339284398E-4,0.0015106559069639339)], [(-0.001370980718313348,-2.4014019974697736E-4), (-3.5628964552930848E-4,-6.542394855209258E-4), (0.0017045839820314776,-0.0032647636846900247)]]> : tensor<2x3xcomplex<f64>>
    %cst_0 = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<complex<f64>>
    return %cst, %cst_0 : tensor<2x3xcomplex<f64>>, tensor<complex<f64>>
  }
  func.func private @expected() -> (tensor<2x1xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[(-0.0010977968750361337,-0.0020786308515255291)], [(-3.5628964552930848E-4,-6.542394855209258E-4)]]> : tensor<2x1xcomplex<f64>>
    return %cst : tensor<2x1xcomplex<f64>>
  }
}
