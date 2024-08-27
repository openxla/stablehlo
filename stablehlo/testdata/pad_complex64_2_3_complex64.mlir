// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<6x4xcomplex<f32>> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<2x3xcomplex<f32>>, tensor<complex<f32>>)
    %1 = call @expected() : () -> tensor<6x4xcomplex<f32>>
    %2 = stablehlo.pad %0#0, %0#1, low = [1, 0], high = [2, 1], interior = [1, 0] : (tensor<2x3xcomplex<f32>>, tensor<complex<f32>>) -> tensor<6x4xcomplex<f32>>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<6x4xcomplex<f32>>, tensor<6x4xcomplex<f32>>) -> ()
    return %2 : tensor<6x4xcomplex<f32>>
  }
  func.func private @inputs() -> (tensor<2x3xcomplex<f32>> {mhlo.layout_mode = "default"}, tensor<complex<f32>> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[(-1.39428914E-4,8.64812405E-4), (-0.00152378506,-0.0014140791), (-0.00155372021,5.68573596E-4)], [(1.94506283E-4,0.00147564197), (-0.00122765638,-2.12210784E-4), (-3.98263692E-5,4.93056548E-4)]]> : tensor<2x3xcomplex<f32>>
    %cst_0 = stablehlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<complex<f32>>
    return %cst, %cst_0 : tensor<2x3xcomplex<f32>>, tensor<complex<f32>>
  }
  func.func private @expected() -> (tensor<6x4xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[(0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00)], [(-1.39428914E-4,8.64812405E-4), (-0.00152378506,-0.0014140791), (-0.00155372021,5.68573596E-4), (0.000000e+00,0.000000e+00)], [(0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00)], [(1.94506283E-4,0.00147564197), (-0.00122765638,-2.12210784E-4), (-3.98263692E-5,4.93056548E-4), (0.000000e+00,0.000000e+00)], [(0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00)], [(0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00)]]> : tensor<6x4xcomplex<f32>>
    return %cst : tensor<6x4xcomplex<f32>>
  }
}
