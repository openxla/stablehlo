// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x6xcomplex<f32>> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<4x3xf64>, tensor<3x6xcomplex<f32>>)
    %1 = call @expected() : () -> tensor<4x6xcomplex<f32>>
    %2 = stablehlo.convert %0#0 : (tensor<4x3xf64>) -> tensor<4x3xcomplex<f32>>
    %3 = stablehlo.convert %0#1 : tensor<3x6xcomplex<f32>>
    %4 = stablehlo.dot_general %2, %3, contracting_dims = [1] x [0] : (tensor<4x3xcomplex<f32>>, tensor<3x6xcomplex<f32>>) -> tensor<4x6xcomplex<f32>>
    stablehlo.custom_call @check.expect_close(%4, %1) {has_side_effect = true} : (tensor<4x6xcomplex<f32>>, tensor<4x6xcomplex<f32>>) -> ()
    return %4 : tensor<4x6xcomplex<f32>>
  }
  func.func private @inputs() -> (tensor<4x3xf64> {mhlo.layout_mode = "default"}, tensor<3x6xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[5.3344602496326452, -3.3473225300748761, -0.045295202073283108], [-0.013782969929730176, 1.2721496518432649, 1.4442056711865696], [-0.61793096410831239, -0.48528855673026416, 2.2336247557287932], [-0.33843513481191584, 6.7866123008317469, 0.77216249934130154]]> : tensor<4x3xf64>
    %cst_0 = stablehlo.constant dense<[[(0.576168597,4.83222532), (-2.9939034,-2.373360e+00), (1.89569354,0.293711483), (-2.17946529,-0.13490212), (5.24347687,2.61133838), (-2.08481026,5.35744476)], [(3.22966337,-1.81060171), (4.97413683,1.96379447), (0.917333841,2.94159937), (1.60617745,0.895324051), (1.06881773,-7.15810061), (0.929969906,6.78598738)], [(1.24720037,0.221818566), (1.01871955,5.96387529), (4.42998266,1.61882496), (-4.24127102,3.19067073), (3.49672914,-2.42325091), (-0.591893554,1.57355595)]]> : tensor<3x6xcomplex<f32>>
    return %cst, %cst_0 : tensor<4x3xf64>, tensor<3x6xcomplex<f32>>
  }
  func.func private @expected() -> (tensor<4x6xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[(-7.79366922,31.8279343), (-32.6670418,-19.5041809), (6.84123325,-8.35301399), (-16.8105545,-3.86109018), (24.2350559,38.000309), (-14.2074375,5.79291439)], [(5.90188789,-2.04960728), (7.84035205,11.1440144), (7.53866386,6.07602262), (-4.05192947,5.748830e+00), (6.33742142,-12.64184), (0.356979787,10.8314886)], [(0.862426519,-1.61185777), (1.71157086,13.8346252), (8.27833939,2.00682926), (-8.9061079,6.77563047), (4.05158901,-3.55251551), (-4.851030e-01,-3.08895969)], [(22.6865196,-13.7519674), (35.5573959,18.7358208), (9.00468635,21.11409), (8.36316108,8.5855894), (8.1791172,-51.3341675), (6.55988073,45.455761)]]> : tensor<4x6xcomplex<f32>>
    return %cst : tensor<4x6xcomplex<f32>>
  }
}
