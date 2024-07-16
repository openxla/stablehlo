// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<3x4xcomplex<f32>> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<3x4xf32>, tensor<3x4xf32>)
    %1 = call @expected() : () -> tensor<3x4xcomplex<f32>>
    %2 = stablehlo.complex %0#0, %0#1 : tensor<3x4xcomplex<f32>>
    stablehlo.custom_call @check.expect_close(%2, %1) {has_side_effect = true} : (tensor<3x4xcomplex<f32>>, tensor<3x4xcomplex<f32>>) -> ()
    return %2 : tensor<3x4xcomplex<f32>>
  }
  func.func private @inputs() -> (tensor<3x4xf32> {mhlo.layout_mode = "default"}, tensor<3x4xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[4.13312864, -3.9565289, 3.49465394, 6.253200e-02], [0.410699964, -1.21260703, 2.13978839, -1.51862335], [1.75706387, -0.269945771, -4.01345253, 4.72941542]]> : tensor<3x4xf32>
    %cst_0 = stablehlo.constant dense<[[-4.41265631, -0.327641249, 2.52980947, -0.798980951], [-2.14335227, 3.32583117, 1.07032835, -5.32339954], [-2.90522337, 5.24038172, -4.99515343, -1.49502265]]> : tensor<3x4xf32>
    return %cst, %cst_0 : tensor<3x4xf32>, tensor<3x4xf32>
  }
  func.func private @expected() -> (tensor<3x4xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[(4.13312864,-4.41265631), (-3.9565289,-0.327641249), (3.49465394,2.52980947), (6.253200e-02,-0.798980951)], [(0.410699964,-2.14335227), (-1.21260703,3.32583117), (2.13978839,1.07032835), (-1.51862335,-5.32339954)], [(1.75706387,-2.90522337), (-0.269945771,5.24038172), (-4.01345253,-4.99515343), (4.72941542,-1.49502265)]]> : tensor<3x4xcomplex<f32>>
    return %cst : tensor<3x4xcomplex<f32>>
  }
}
