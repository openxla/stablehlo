// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x6xcomplex<f32>> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<4x3xf16>, tensor<3x6xcomplex<f32>>)
    %1 = call @expected() : () -> tensor<4x6xcomplex<f32>>
    %2 = stablehlo.convert %0#0 : (tensor<4x3xf16>) -> tensor<4x3xcomplex<f32>>
    %3 = stablehlo.convert %0#1 : tensor<3x6xcomplex<f32>>
    %4 = stablehlo.dot_general %2, %3, contracting_dims = [1] x [0] : (tensor<4x3xcomplex<f32>>, tensor<3x6xcomplex<f32>>) -> tensor<4x6xcomplex<f32>>
    stablehlo.custom_call @check.expect_close(%4, %1) {has_side_effect = true} : (tensor<4x6xcomplex<f32>>, tensor<4x6xcomplex<f32>>) -> ()
    return %4 : tensor<4x6xcomplex<f32>>
  }
  func.func private @inputs() -> (tensor<4x3xf16> {mhlo.layout_mode = "default"}, tensor<3x6xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[-2.324220e+00, 4.880370e-01, -3.251950e-01], [-4.117190e+00, 5.976560e+00, -3.890630e+00], [-2.333980e+00, -3.896480e+00, -1.708980e-01], [-3.318360e+00, 9.296870e+00, 5.988280e+00]]> : tensor<4x3xf16>
    %cst_0 = stablehlo.constant dense<[[(-0.843801319,4.36061573), (1.59704077,0.665311217), (5.54299736,6.720150e-01), (-0.952389836,0.622085988), (-0.795956373,-1.01807463), (-1.65721118,0.190785199)], [(-0.199466407,-2.12686372), (-3.70492983,-1.39143848), (3.93882036,2.21310306), (-4.12120438,-3.86826062), (0.326258302,0.981736123), (-1.57882833,-3.43677235)], [(0.968069911,2.14947581), (3.07858276,-0.00304714404), (-2.26005793,5.24462223), (-4.07814837,5.14572525), (-1.36028254,-1.94103074), (-3.80221438,5.37264967)]]> : tensor<3x6xcomplex<f32>>
    return %cst, %cst_0 : tensor<4x3xf16>, tensor<3x6xcomplex<f32>>
  }
  func.func private @expected() -> (tensor<4x6xcomplex<f32>> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<[[(1.549020e+00,-11.8720131), (-6.52115583,-2.22441149), (-10.2258873,-2.187360e+00), (1.52845645,-5.00708437), (2.4515605,3.47656584), (4.31765652,-3.86785936)], [(-1.48443222,-39.0276108), (-40.6956711,-11.043375), (9.51208305,-9.94492149), (-4.84292316,-45.7002335), (10.5193539,17.6108341), (12.1800728,-42.2285461)], [(2.58119535,-2.257660e+00), (10.1826077,3.8694129), (-27.8985806,-11.0880919), (18.9780197,12.7412825), (0.818959475,-1.117430e+00), (10.6695776,12.0278625)], [(6.74269676,-21.3716087), (-21.3084087,-15.1620188), (4.69120121,49.7512283), (-59.5750504,-7.2129879), (-2.47130299,0.881977081), (-31.9476757,-0.411399841)]]> : tensor<4x6xcomplex<f32>>
    return %cst : tensor<4x6xcomplex<f32>>
  }
}
