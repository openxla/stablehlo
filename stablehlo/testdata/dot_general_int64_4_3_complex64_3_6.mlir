// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x6xcomplex<f32>> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<4x3xi64>, tensor<3x6xcomplex<f32>>)
    %1 = call @expected() : () -> tensor<4x6xcomplex<f32>>
    %2 = stablehlo.convert %0#0 : (tensor<4x3xi64>) -> tensor<4x3xcomplex<f32>>
    %3 = stablehlo.convert %0#1 : tensor<3x6xcomplex<f32>>
    %4 = stablehlo.dot_general %2, %3, contracting_dims = [1] x [0] : (tensor<4x3xcomplex<f32>>, tensor<3x6xcomplex<f32>>) -> tensor<4x6xcomplex<f32>>
    stablehlo.custom_call @check.expect_close(%4, %1) {has_side_effect = true} : (tensor<4x6xcomplex<f32>>, tensor<4x6xcomplex<f32>>) -> ()
    return %4 : tensor<4x6xcomplex<f32>>
  }
  func.func private @inputs() -> (tensor<4x3xi64> {mhlo.layout_mode = "default"}, tensor<3x6xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %c = stablehlo.constant dense<[[-1, -3, 0], [2, 1, -2], [0, -1, 4], [-1, -5, -1]]> : tensor<4x3xi64>
    %cst = stablehlo.constant dense<[[(2.83740211,-5.58472157), (4.0482688,0.876852869), (-0.0827702209,7.11556339), (-0.842830538,1.60649145), (1.28883183,-3.56874871), (-4.55080938,0.913843214)], [(-2.69412303,1.50316823), (0.464145482,-5.87185955), (-1.02071273,3.80705047), (3.27020288,0.340390116), (1.05008376,-1.85043538), (-2.64531517,2.74554229)], [(2.29520464,-2.21171331), (0.28629294,-0.0440344475), (4.38808918,-2.23752379), (3.23008013,-0.31844461), (2.5760622,-2.5523777), (3.08757639,2.03543687)]]> : tensor<3x6xcomplex<f32>>
    return %c, %cst : tensor<4x3xi64>, tensor<3x6xcomplex<f32>>
  }
  func.func private @expected() -> (tensor<4x6xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[(5.24496651,1.07521677), (-5.4407053,16.7387257), (3.14490819,-18.5367146), (-8.9677782,-2.62766171), (-4.4390831,9.12005519), (12.4867554,-9.15046977)], [(-1.6097281,-5.2428484), (7.98809719,-4.03008461), (-9.9624319,22.5132256), (-4.87561846,4.19026232), (-1.52437687,-3.88317776), (-17.9220867,0.502355099)], [(11.8749418,-10.3500214), (0.681026279,5.69572163), (18.5730686,-12.7571459), (9.65011787,-1.61416852), (9.25416469,-8.35907554), (14.9956207,5.39620495)], [(8.33800888,0.280593872), (-6.65528917,28.5264816), (0.798244953,-23.91329), (-18.7382641,-2.98999739), (-9.11531257,15.3733034), (14.6898098,-16.6769924)]]> : tensor<4x6xcomplex<f32>>
    return %cst : tensor<4x6xcomplex<f32>>
  }
}
