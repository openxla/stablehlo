// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3x3xcomplex<f64>> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<3x3xcomplex<f64>>, tensor<3x3xcomplex<f64>>)
    %1 = call @expected() : () -> tensor<3x3xcomplex<f64>>
    %2 = stablehlo.real %0#0 : (tensor<3x3xcomplex<f64>>) -> tensor<3x3xf64>
    %3 = stablehlo.real %0#1 : (tensor<3x3xcomplex<f64>>) -> tensor<3x3xf64>
    %4 = stablehlo.compare  EQ, %2, %3,  FLOAT : (tensor<3x3xf64>, tensor<3x3xf64>) -> tensor<3x3xi1>
    %5 = stablehlo.compare  LT, %2, %3,  FLOAT : (tensor<3x3xf64>, tensor<3x3xf64>) -> tensor<3x3xi1>
    %6 = stablehlo.imag %0#0 : (tensor<3x3xcomplex<f64>>) -> tensor<3x3xf64>
    %7 = stablehlo.imag %0#1 : (tensor<3x3xcomplex<f64>>) -> tensor<3x3xf64>
    %8 = stablehlo.compare  LT, %6, %7,  FLOAT : (tensor<3x3xf64>, tensor<3x3xf64>) -> tensor<3x3xi1>
    %9 = stablehlo.select %4, %8, %5 : tensor<3x3xi1>, tensor<3x3xi1>
    %10 = stablehlo.select %9, %0#0, %0#1 : tensor<3x3xi1>, tensor<3x3xcomplex<f64>>
    stablehlo.custom_call @check.expect_close(%10, %1) {has_side_effect = true} : (tensor<3x3xcomplex<f64>>, tensor<3x3xcomplex<f64>>) -> ()
    return %10 : tensor<3x3xcomplex<f64>>
  }
  func.func private @inputs() -> (tensor<3x3xcomplex<f64>> {mhlo.layout_mode = "default"}, tensor<3x3xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[(0x7FF8000000000000,0.000000e+00), (0x7FF8000000000000,0.000000e+00), (0x7FF8000000000000,0.000000e+00)], [(0x7FF0000000000000,0.000000e+00), (0x7FF0000000000000,0.000000e+00), (0x7FF0000000000000,0.000000e+00)], [(0xFFF0000000000000,0.000000e+00), (0xFFF0000000000000,0.000000e+00), (0xFFF0000000000000,0.000000e+00)]]> : tensor<3x3xcomplex<f64>>
    %cst_0 = stablehlo.constant dense<[[(0x7FF8000000000000,0.000000e+00), (0x7FF0000000000000,0.000000e+00), (0xFFF0000000000000,0.000000e+00)], [(0x7FF8000000000000,0.000000e+00), (0x7FF0000000000000,0.000000e+00), (0xFFF0000000000000,0.000000e+00)], [(0x7FF8000000000000,0.000000e+00), (0x7FF0000000000000,0.000000e+00), (0xFFF0000000000000,0.000000e+00)]]> : tensor<3x3xcomplex<f64>>
    return %cst, %cst_0 : tensor<3x3xcomplex<f64>>, tensor<3x3xcomplex<f64>>
  }
  func.func private @expected() -> (tensor<3x3xcomplex<f64>> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[(0x7FF8000000000000,0.000000e+00), (0x7FF0000000000000,0.000000e+00), (0xFFF0000000000000,0.000000e+00)], [(0x7FF8000000000000,0.000000e+00), (0x7FF0000000000000,0.000000e+00), (0xFFF0000000000000,0.000000e+00)], [(0x7FF8000000000000,0.000000e+00), (0xFFF0000000000000,0.000000e+00), (0xFFF0000000000000,0.000000e+00)]]> : tensor<3x3xcomplex<f64>>
    return %cst : tensor<3x3xcomplex<f64>>
  }
}
