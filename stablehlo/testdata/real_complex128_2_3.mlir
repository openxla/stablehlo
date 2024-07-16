// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<2x3xf64> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<2x3xcomplex<f64>>
    %1 = call @expected() : () -> tensor<2x3xf64>
    %2 = stablehlo.real %0 : (tensor<2x3xcomplex<f64>>) -> tensor<2x3xf64>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<2x3xf64>, tensor<2x3xf64>) -> ()
    return %2 : tensor<2x3xf64>
  }
  func.func private @inputs() -> (tensor<2x3xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[(-1.6158824351682006,0.133493958338603), (3.6881363830118095,-0.83637693079356756), (-5.3928605551928701,4.2108833186663865)], [(0.92740008249534012,0.82483512823598004), (0.85505669370970705,-6.4405396519565743), (-0.065481723763739591,2.5754166605013493)]]> : tensor<2x3xcomplex<f64>>
    return %cst : tensor<2x3xcomplex<f64>>
  }
  func.func private @expected() -> (tensor<2x3xf64> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-1.6158824351682006, 3.6881363830118095, -5.3928605551928701], [0.92740008249534012, 0.85505669370970705, -0.065481723763739591]]> : tensor<2x3xf64>
    return %cst : tensor<2x3xf64>
  }
}
