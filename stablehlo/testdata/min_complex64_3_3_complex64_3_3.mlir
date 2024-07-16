// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3x3xcomplex<f32>> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<3x3xcomplex<f32>>, tensor<3x3xcomplex<f32>>)
    %1 = call @expected() : () -> tensor<3x3xcomplex<f32>>
    %2 = stablehlo.real %0#0 : (tensor<3x3xcomplex<f32>>) -> tensor<3x3xf32>
    %3 = stablehlo.real %0#1 : (tensor<3x3xcomplex<f32>>) -> tensor<3x3xf32>
    %4 = stablehlo.compare  EQ, %2, %3,  FLOAT : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x3xi1>
    %5 = stablehlo.compare  LT, %2, %3,  FLOAT : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x3xi1>
    %6 = stablehlo.imag %0#0 : (tensor<3x3xcomplex<f32>>) -> tensor<3x3xf32>
    %7 = stablehlo.imag %0#1 : (tensor<3x3xcomplex<f32>>) -> tensor<3x3xf32>
    %8 = stablehlo.compare  LT, %6, %7,  FLOAT : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<3x3xi1>
    %9 = stablehlo.select %4, %8, %5 : tensor<3x3xi1>, tensor<3x3xi1>
    %10 = stablehlo.select %9, %0#0, %0#1 : tensor<3x3xi1>, tensor<3x3xcomplex<f32>>
    stablehlo.custom_call @check.expect_close(%10, %1) {has_side_effect = true} : (tensor<3x3xcomplex<f32>>, tensor<3x3xcomplex<f32>>) -> ()
    return %10 : tensor<3x3xcomplex<f32>>
  }
  func.func private @inputs() -> (tensor<3x3xcomplex<f32>> {mhlo.layout_mode = "default"}, tensor<3x3xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[(0x7FC00000,0.000000e+00), (0x7FC00000,0.000000e+00), (0x7FC00000,0.000000e+00)], [(0x7F800000,0.000000e+00), (0x7F800000,0.000000e+00), (0x7F800000,0.000000e+00)], [(0xFF800000,0.000000e+00), (0xFF800000,0.000000e+00), (0xFF800000,0.000000e+00)]]> : tensor<3x3xcomplex<f32>>
    %cst_0 = stablehlo.constant dense<[[(0x7FC00000,0.000000e+00), (0x7F800000,0.000000e+00), (0xFF800000,0.000000e+00)], [(0x7FC00000,0.000000e+00), (0x7F800000,0.000000e+00), (0xFF800000,0.000000e+00)], [(0x7FC00000,0.000000e+00), (0x7F800000,0.000000e+00), (0xFF800000,0.000000e+00)]]> : tensor<3x3xcomplex<f32>>
    return %cst, %cst_0 : tensor<3x3xcomplex<f32>>, tensor<3x3xcomplex<f32>>
  }
  func.func private @expected() -> (tensor<3x3xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[(0x7FC00000,0.000000e+00), (0x7F800000,0.000000e+00), (0xFF800000,0.000000e+00)], [(0x7FC00000,0.000000e+00), (0x7F800000,0.000000e+00), (0xFF800000,0.000000e+00)], [(0x7FC00000,0.000000e+00), (0xFF800000,0.000000e+00), (0xFF800000,0.000000e+00)]]> : tensor<3x3xcomplex<f32>>
    return %cst : tensor<3x3xcomplex<f32>>
  }
}
