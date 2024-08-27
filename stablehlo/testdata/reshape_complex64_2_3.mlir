// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3x2xcomplex<f32>> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = call @inputs() : () -> tensor<2x3xcomplex<f32>>
    %1 = call @expected() : () -> tensor<3x2xcomplex<f32>>
    %2 = stablehlo.reshape %0 : (tensor<2x3xcomplex<f32>>) -> tensor<3x2xcomplex<f32>>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<3x2xcomplex<f32>>, tensor<3x2xcomplex<f32>>) -> ()
    return %2 : tensor<3x2xcomplex<f32>>
  }
  func.func private @inputs() -> (tensor<2x3xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[(0.617260575,-0.0616655722), (-5.14312601,0.740792274), (-8.860930e+00,-2.22399831)], [(0.265819639,2.97546291), (1.72794175,7.77228832), (0.378611773,0.622524082)]]> : tensor<2x3xcomplex<f32>>
    return %cst : tensor<2x3xcomplex<f32>>
  }
  func.func private @expected() -> (tensor<3x2xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[(0.617260575,-0.0616655722), (-5.14312601,0.740792274)], [(-8.860930e+00,-2.22399831), (0.265819639,2.97546291)], [(1.72794175,7.77228832), (0.378611773,0.622524082)]]> : tensor<3x2xcomplex<f32>>
    return %cst : tensor<3x2xcomplex<f32>>
  }
}
