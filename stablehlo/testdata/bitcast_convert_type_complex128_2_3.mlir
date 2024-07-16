// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<2x3xcomplex<f64>> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<2x3xcomplex<f64>>
    %1 = call @expected() : () -> tensor<2x3xcomplex<f64>>
    %2 = stablehlo.bitcast_convert %0 : (tensor<2x3xcomplex<f64>>) -> tensor<2x3xcomplex<f64>>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<2x3xcomplex<f64>>, tensor<2x3xcomplex<f64>>) -> ()
    return %2 : tensor<2x3xcomplex<f64>>
  }
  func.func private @inputs() -> (tensor<2x3xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[(-1.0188829069777727,-1.7340762688128248), (-0.52944611775003347,2.8218267477233883), (-2.8165229775578471,3.3006773630951018)], [(0.77793671976434475,0.39654971695575442), (2.1567318771649968,-1.1468829935302329), (-2.6251103026761804,1.221256199025839)]]> : tensor<2x3xcomplex<f64>>
    return %cst : tensor<2x3xcomplex<f64>>
  }
  func.func private @expected() -> (tensor<2x3xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[(-1.0188829069777727,-1.7340762688128248), (-0.52944611775003347,2.8218267477233883), (-2.8165229775578471,3.3006773630951018)], [(0.77793671976434475,0.39654971695575442), (2.1567318771649968,-1.1468829935302329), (-2.6251103026761804,1.221256199025839)]]> : tensor<2x3xcomplex<f64>>
    return %cst : tensor<2x3xcomplex<f64>>
  }
}
