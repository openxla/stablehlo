// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x6xcomplex<f32>> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<4x3xbf16>, tensor<3x6xcomplex<f32>>)
    %1 = call @expected() : () -> tensor<4x6xcomplex<f32>>
    %2 = stablehlo.convert %0#0 : (tensor<4x3xbf16>) -> tensor<4x3xcomplex<f32>>
    %3 = stablehlo.convert %0#1 : tensor<3x6xcomplex<f32>>
    %4 = stablehlo.dot_general %2, %3, contracting_dims = [1] x [0] : (tensor<4x3xcomplex<f32>>, tensor<3x6xcomplex<f32>>) -> tensor<4x6xcomplex<f32>>
    stablehlo.custom_call @check.expect_close(%4, %1) {has_side_effect = true} : (tensor<4x6xcomplex<f32>>, tensor<4x6xcomplex<f32>>) -> ()
    return %4 : tensor<4x6xcomplex<f32>>
  }
  func.func private @inputs() -> (tensor<4x3xbf16> {mhlo.layout_mode = "default"}, tensor<3x6xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-1.203130e+00, 3.281250e+00, -2.062500e+00], [-1.648440e+00, 2.218750e+00, 1.304690e+00], [-1.140630e+00, 2.406250e+00, 2.636720e-01], [-1.210940e+00, 9.257810e-01, -1.671880e+00]]> : tensor<4x3xbf16>
    %cst_0 = stablehlo.constant dense<[[(-1.23465157,-1.97534931), (1.670120e+00,0.216153592), (-5.166550e+00,-0.334587127), (0.912404239,0.554522038), (4.53575134,-0.999703705), (-0.30895409,-1.16958821)], [(-0.4028036,-2.49677873), (-3.14465165,-2.28515124), (0.610261738,7.095100e+00), (-2.31424093,1.29751551), (0.101109952,2.62985325), (-1.06779826,5.842440e+00)], [(1.69907737,3.1651814), (0.615705729,-2.47314763), (-0.0892849043,0.24179633), (-0.259796768,1.7544111), (-4.16017103,-3.20194054), (0.222340554,-2.38368416)]]> : tensor<3x6xcomplex<f32>>
    return %cst, %cst_0 : tensor<4x3xbf16>, tensor<3x6xcomplex<f32>>
  }
  func.func private @expected() -> (tensor<4x6xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[(-3.34060621,-12.3441505), (-13.5976439,-2.65734577), (8.4025774,23.1846409), (-8.15550899,-0.0281591415), (3.45504379,16.435976), (-3.59058022,25.4940147)], [(3.3582902,1.84608459), (-8.926980e+00,-8.653180e+00), (9.75426483,16.6092663), (-6.97771692,4.25372601), (-12.6802883,3.30540419), (-1.56980085,11.7809448)], [(0.887027144,-2.92017174), (-9.30945491,-6.397295), (7.33799696,17.5179768), (-6.67785406,2.95223379), (-6.02721596,6.62410975), (-2.15836382,14.7639227)], [(-1.71846724,-5.21123409), (-5.9630537,1.75749493), (6.97061157,6.56942177), (-2.81299782,-2.40343189), (1.55638027,8.99849128), (-0.986149132,10.8103418)]]> : tensor<4x6xcomplex<f32>>
    return %cst : tensor<4x6xcomplex<f32>>
  }
}
