// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<3x5x7xf32>, tensor<2x4x6xf32>)
    %1 = call @expected() : () -> tensor<2x4x6xf32>
    %2 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %3 = stablehlo.pad %0#1, %2, low = [1, 1, 1], high = [1, 1, 1], interior = [0, 0, 0] : (tensor<2x4x6xf32>, tensor<f32>) -> tensor<4x6x8xf32>
    %4 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %5 = "stablehlo.select_and_scatter"(%3, %0#0, %4) ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %8 = stablehlo.compare  GE, %arg0, %arg1,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
      stablehlo.return %8 : tensor<i1>
    }, {
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %8 = stablehlo.add %arg0, %arg1 : tensor<f32>
      stablehlo.return %8 : tensor<f32>
    }) {window_dimensions = dense<2> : tensor<3xi64>} : (tensor<4x6x8xf32>, tensor<3x5x7xf32>, tensor<f32>) -> tensor<4x6x8xf32>
    %6 = "stablehlo.slice"(%5) {limit_indices = dense<[3, 5, 7]> : tensor<3xi64>, start_indices = dense<1> : tensor<3xi64>, strides = dense<1> : tensor<3xi64>} : (tensor<4x6x8xf32>) -> tensor<2x4x6xf32>
    %7 = stablehlo.custom_call @check.eq(%6, %1) : (tensor<2x4x6xf32>, tensor<2x4x6xf32>) -> tensor<i1>
    return %7 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<3x5x7xf32>, tensor<2x4x6xf32>) {
    %0 = stablehlo.constant dense<"0x426C42C0DB2CC3BFF8E0B6BB5BDD01C142A28EBE9283213E358764C057B007C0C342D63F91D74E40B0D9A3BDBFAB643FDFB8C4BFF62D17C038B4C43E5079A1BD3EC1C13F1B068B3F5F485BBF29C777C08F95A740C13151406E12B9BF479E013FCDC5953D8F924EBF76D5AB40034AE9BFA6573240277D8DC096CD473F99F31DBF5FC213C09F9945C06CA04AC0029B48C0B635A1C0F65163C0A00FBBBFCB978B3FC9D2AA409915A5BFCFA49740EEB83BC08FCFA9BFBFB0903F164E1540EB8169C09415DFBF77207BBF923B7AC0887D6FC084A00F40502A32BFED45BDBFFBF2EBBF40FB44C04189BDBFC351E6BE3BB3AB40FEB384C0E6E3C2BFA9CF64C0983B0541300CC73F92758A3FB62005C1310F3B3F7A03783FD94828BFF67E17C001112C406708BF3F21F34BC0B32685C08395B1C0F14E90BFD9DBE43F891DC940021A30404C196C405C2AC83F8A215A3F4FCD8E4036E5B04021F18EC086762ABF1578A43F3839FC3F4C6B04BF874092BFAD67894036D2E1C0CBB398BE5017C9BF382EB33FDC02203EB7C09C3ED4C4BABF413CC340A14F8FC0F62823C0160BC2BF971CAEC08AD13440"> : tensor<3x5x7xf32>
    %1 = stablehlo.constant dense<[[[1.23659277, 0.167427897, 1.66209292, -0.445933163, 1.30365491, 1.48936605], [-0.381028771, 1.46708107, 0.561221182, -9.44773292, 2.2169826, -0.633117139], [1.16430485, 0.543411791, 0.246518821, 2.4222579, -1.58968532, -0.530795813], [-2.70711923, -2.38582158, 0.044348821, 1.30091822, 1.47974443, -3.22229099]], [[4.104470e+00, 0.550411761, -2.16386747, 0.315645754, -1.1609273, -3.57031155], [4.31706285, -0.948032677, -1.71081841, -4.81900358, 0.579106927, -1.17353725], [-1.91352499, -0.394621432, -4.42012691, 3.47869635, -0.68511641, -1.02591705], [-0.00396529213, -1.14221811, -0.786569714, 1.12769604, -1.54288232, 1.0621357]]]> : tensor<2x4x6xf32>
    return %0, %1 : tensor<3x5x7xf32>, tensor<2x4x6xf32>
  }
  func.func private @expected() -> tensor<2x4x6xf32> {
    %0 = stablehlo.constant dense<[[[-6.6828022, 0.000000e+00, -10.1797657, 0.000000e+00, 0.811988174, -3.46954107], [0.000000e+00, -0.633259296, 0.000000e+00, 0.000000e+00, -7.30953646, 0.000000e+00], [-2.35161209, 0.0564788282, 0.000000e+00, -0.504241824, 0.000000e+00, 1.57108271], [2.78659964, -4.42152739, 1.8621937, -8.93748474, 0.150491178, -3.16604137]], [[-7.85085487, 4.24403143, 0.000000e+00, -3.65864468, -5.54950094, -1.1274091], [6.04871511, 0.000000e+00, 0.000000e+00, 0.000000e+00, 1.89860463, 4.46256208], [0.000000e+00, -0.964117765, 0.000000e+00, 5.85110569, 0.000000e+00, -1.14259422], [11.7610769, 0.000000e+00, -4.47847033, -4.06533813, 0.000000e+00, -6.385818]]]> : tensor<2x4x6xf32>
    return %0 : tensor<2x4x6xf32>
  }
}

