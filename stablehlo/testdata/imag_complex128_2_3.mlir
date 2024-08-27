// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<2x3xf64> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<2x3xcomplex<f64>>
    %1 = call @expected() : () -> tensor<2x3xf64>
    %2 = stablehlo.imag %0 : (tensor<2x3xcomplex<f64>>) -> tensor<2x3xf64>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<2x3xf64>, tensor<2x3xf64>) -> ()
    return %2 : tensor<2x3xf64>
  }
  func.func private @inputs() -> (tensor<2x3xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[(-2.9861362918419676,-2.2613215132343165), (-0.055134557576814597,-4.0719526661776255), (-0.63150008801334312,-0.02396522156877821)], [(-1.7564080570937262,6.0012926290220046), (3.0963484828247498,0.97704345195118303), (2.7687030824134569,3.1974725722033863)]]> : tensor<2x3xcomplex<f64>>
    return %cst : tensor<2x3xcomplex<f64>>
  }
  func.func private @expected() -> (tensor<2x3xf64> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-2.2613215132343165, -4.0719526661776255, -0.02396522156877821], [6.0012926290220046, 0.97704345195118303, 3.1974725722033863]]> : tensor<2x3xf64>
    return %cst : tensor<2x3xf64>
  }
}
