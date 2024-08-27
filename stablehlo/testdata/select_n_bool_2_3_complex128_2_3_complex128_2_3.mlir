// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<2x3xcomplex<f64>> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:3 = call @inputs() : () -> (tensor<2x3xi1>, tensor<2x3xcomplex<f64>>, tensor<2x3xcomplex<f64>>)
    %1 = call @expected() : () -> tensor<2x3xcomplex<f64>>
    %2 = stablehlo.select %0#0, %0#2, %0#1 : tensor<2x3xi1>, tensor<2x3xcomplex<f64>>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<2x3xcomplex<f64>>, tensor<2x3xcomplex<f64>>) -> ()
    return %2 : tensor<2x3xcomplex<f64>>
  }
  func.func private @inputs() -> (tensor<2x3xi1> {mhlo.layout_mode = "default"}, tensor<2x3xcomplex<f64>> {mhlo.layout_mode = "default"}, tensor<2x3xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<true> : tensor<2x3xi1>
    %cst = stablehlo.constant dense<[[(0.20488651448381795,-0.98189562308349143), (-0.34537618521175339,-3.5498123654189522), (3.4083892023097366,-0.022663425961758327)], [(3.1058804720301971,-0.915100536489034), (-3.2222121641768533,1.3265816258718024), (2.6933434704288999,0.72250984287954856)]]> : tensor<2x3xcomplex<f64>>
    %cst_0 = stablehlo.constant dense<[[(-0.28229432556002887,0.71646410433306551), (-0.76022405473318333,0.55355351453915591), (-0.50504892626084841,-2.0612439610208977)], [(3.0892378364363644,-2.5828326941290856), (-1.5486345575155698,-0.32902175096594749), (-0.77443543516507185,3.6713526992361958)]]> : tensor<2x3xcomplex<f64>>
    return %c, %cst, %cst_0 : tensor<2x3xi1>, tensor<2x3xcomplex<f64>>, tensor<2x3xcomplex<f64>>
  }
  func.func private @expected() -> (tensor<2x3xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[(-0.28229432556002887,0.71646410433306551), (-0.76022405473318333,0.55355351453915591), (-0.50504892626084841,-2.0612439610208977)], [(3.0892378364363644,-2.5828326941290856), (-1.5486345575155698,-0.32902175096594749), (-0.77443543516507185,3.6713526992361958)]]> : tensor<2x3xcomplex<f64>>
    return %cst : tensor<2x3xcomplex<f64>>
  }
}
