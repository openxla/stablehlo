// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3x2xcomplex<f32>> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<3x2xf32>, tensor<3x1xf32>)
    %1 = call @expected() : () -> tensor<3x2xcomplex<f32>>
    %2 = stablehlo.broadcast_in_dim %0#1, dims = [0, 1] : (tensor<3x1xf32>) -> tensor<3x2xf32>
    %3 = stablehlo.complex %0#0, %2 : tensor<3x2xcomplex<f32>>
    stablehlo.custom_call @check.expect_close(%3, %1) {has_side_effect = true} : (tensor<3x2xcomplex<f32>>, tensor<3x2xcomplex<f32>>) -> ()
    return %3 : tensor<3x2xcomplex<f32>>
  }
  func.func private @inputs() -> (tensor<3x2xf32> {mhlo.layout_mode = "default"}, tensor<3x1xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[4.3779788, 0.256987363], [-1.94227421, -0.415562898], [-4.24635506, 0.812674164]]> : tensor<3x2xf32>
    %cst_0 = stablehlo.constant dense<[[-1.35233212], [-1.11680901], [-1.31206405]]> : tensor<3x1xf32>
    return %cst, %cst_0 : tensor<3x2xf32>, tensor<3x1xf32>
  }
  func.func private @expected() -> (tensor<3x2xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[(4.3779788,-1.35233212), (0.256987363,-1.35233212)], [(-1.94227421,-1.11680901), (-0.415562898,-1.11680901)], [(-4.24635506,-1.31206405), (0.812674164,-1.31206405)]]> : tensor<3x2xcomplex<f32>>
    return %cst : tensor<3x2xcomplex<f32>>
  }
}
