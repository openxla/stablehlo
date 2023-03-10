// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<4x6xcomplex<f32>>
    %1 = call @expected() : () -> tensor<3x5xcomplex<f32>>
    %2 = stablehlo.constant dense<(1.000000e+00,0.000000e+00)> : tensor<complex<f32>>
    %3 = "stablehlo.reduce_window"(%0, %2) ({
    ^bb0(%arg0: tensor<complex<f32>>, %arg1: tensor<complex<f32>>):
      %5 = stablehlo.multiply %arg0, %arg1 : tensor<complex<f32>>
      stablehlo.return %5 : tensor<complex<f32>>
    }) {window_dimensions = dense<2> : tensor<2xi64>} : (tensor<4x6xcomplex<f32>>, tensor<complex<f32>>) -> tensor<3x5xcomplex<f32>>
    %4 = stablehlo.custom_call @check.eq(%3, %1) : (tensor<3x5xcomplex<f32>>, tensor<3x5xcomplex<f32>>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> tensor<4x6xcomplex<f32>> {
    %0 = stablehlo.constant dense<[[(-9.899160e-02,-2.43280554), (-3.55937552,4.24076939), (-1.68072701,0.818909823), (-1.95779932,2.12700987), (1.61286581,-0.104115866), (3.6997335,0.219332919)], [(7.59286165,0.547871172), (2.84281945,-4.16730499), (0.763019204,0.0755063072), (1.95146608,-2.51156902), (-4.27698946,2.21741581), (1.49481142,-3.11168575)], [(0.566726089,-0.873029828), (-5.01771545,1.18283749), (-2.58140278,-0.589002669), (0.642989814,-0.485557735), (1.77005553,-2.70839953), (-2.86602473,-0.518723965)], [(-3.56504393,1.4034487), (-4.75717258,4.69429779), (4.19491768,0.169697613), (1.58051991,-5.793550e-01), (1.93572891,4.92316675), (-10.0872955,-0.678436934)]]> : tensor<4x6xcomplex<f32>>
    return %0 : tensor<4x6xcomplex<f32>>
  }
  func.func private @expected() -> tensor<3x5xcomplex<f32>> {
    %0 = stablehlo.constant dense<[[(502.536591,-124.316643), (-23.5433674,-32.3840752), (-6.56063604,-11.4320221), (-46.613205,-5.433980e+01), (3.55733323,99.5575942)], [(108.728577,175.040878), (33.6118774,-40.714962), (-1.71896231,4.91049862), (39.6854019,4.55675173), (-117.05423,-104.217117)], [(99.4695434,94.7843704), (-281.411346,259.800812), (-11.2011042,10.0924788), (16.2753143,-16.5544872), (453.732758,219.429657)]]> : tensor<3x5xcomplex<f32>>
    return %0 : tensor<3x5xcomplex<f32>>
  }
}

