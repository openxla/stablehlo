// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3x2xcomplex<f32>> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<2x3xcomplex<f32>>
    %1 = call @expected() : () -> tensor<3x2xcomplex<f32>>
    %2 = stablehlo.transpose %0, dims = [1, 0] : (tensor<2x3xcomplex<f32>>) -> tensor<3x2xcomplex<f32>>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<3x2xcomplex<f32>>, tensor<3x2xcomplex<f32>>) -> ()
    return %2 : tensor<3x2xcomplex<f32>>
  }
  func.func private @inputs() -> (tensor<2x3xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[(-5.08990383,2.25510144), (-5.3642478,-0.41310975), (-0.010698243,-0.417641759)], [(-5.15686893,-8.785930e-01), (2.19871163,-1.17818773), (-5.12186956,2.18995166)]]> : tensor<2x3xcomplex<f32>>
    return %cst : tensor<2x3xcomplex<f32>>
  }
  func.func private @expected() -> (tensor<3x2xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[(-5.08990383,2.25510144), (-5.15686893,-8.785930e-01)], [(-5.3642478,-0.41310975), (2.19871163,-1.17818773)], [(-0.010698243,-0.417641759), (-5.12186956,2.18995166)]]> : tensor<3x2xcomplex<f32>>
    return %cst : tensor<3x2xcomplex<f32>>
  }
}
