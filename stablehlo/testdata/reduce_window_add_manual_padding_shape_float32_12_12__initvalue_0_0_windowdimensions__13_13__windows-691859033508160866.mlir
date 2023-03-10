// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<12x12xf32>
    %1 = call @expected() : () -> tensor<3x2xf32>
    %2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f32>) -> tensor<f32>
    %4 = "stablehlo.reduce_window"(%0, %3) ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %6 = stablehlo.add %arg0, %arg1 : tensor<f32>
      stablehlo.return %6 : tensor<f32>
    }) {padding = dense<[[5, 6], [3, 4]]> : tensor<2x2xi64>, window_dimensions = dense<13> : tensor<2xi64>, window_strides = dense<[5, 6]> : tensor<2xi64>} : (tensor<12x12xf32>, tensor<f32>) -> tensor<3x2xf32>
    %5 = stablehlo.custom_call @check.eq(%4, %1) : (tensor<3x2xf32>, tensor<3x2xf32>) -> tensor<i1>
    return %5 : tensor<i1>
  }
  func.func private @inputs() -> tensor<12x12xf32> {
    %0 = stablehlo.constant dense<"0x297483BF2BA3E1BFFD50AEC065F29040294B5B3F1257B5C0C13F7ABF5519854039D0A44015A2CB3F3D94883FE4C892BEE554BFC0D4630540F1FBCA3DC09F9D40ED6084BFB83D823F08DFE43F36EFC5BF60B6A1C023A84ABF1194A5403CE1D3BEFE9BA53FC018A9BF8840BC3F33A28A403124DCC09A87F73FE1CD933E5532E43FE6099F407857693F2DFD7340CC419DBFCE7E4EC0823D2DC0E5A380BF1C724BC0F7B4B0C078664640F8F04E40816CC1C09EDC853FC4ACB9408B7A0C3FFD57CB3EDEF14BC09F28DEBF1BD806BFFC7DA7408A76C63D9E94AA3F2E0818C0749C063FF5B5233FB464AFC0037897BF3EFB6BC0E45624BFA9F7CEBF807E78C0347F2F3F818E893F974547C0AD720BC0144C2E3E74A3D8BF6570EDBE7FD4E8BF790D193DE4FB163F17276A3FE4D769C0952E28C0C76449BF40FAAF40ECDB23C04C38C73F8891733F8093843F14D6B440B74519C010AD13C090881BBF1D1B603F65172B4081DD9FC084D14F40DA5A8840EAABE53F3C273740CC909640757B7DC01C5A0F409647B43F9D8DC140387D6EC0867E59409E8D843FC5B105C04092BB3F2459FFBF3F821B3F6797C03FBC0347405000A64053790CC05B0A34C01547DE3FFFC395C00FA59840310377BE9FF92D3EFDF939BF0909A7BFABD77D40BA58AC3F0896A4C005401340F52D303F444E46C0BFAE7A3EBC74A3402B69283F0F22DABE0D0BC13F84A80A40A566BF3F74FC16C08670CDC061A68ABF911D70BFC0DA83C05EEB1F40351D14C0336DFFBED51CE9C086E337BED3122140F7FAAFBF205FF2BFADED01C0"> : tensor<12x12xf32>
    return %0 : tensor<12x12xf32>
  }
  func.func private @expected() -> tensor<3x2xf32> {
    %0 = stablehlo.constant dense<[[-6.51430702, 30.7468395], [-2.30757236, 32.5431137], [5.97294712, 13.6468849]]> : tensor<3x2xf32>
    return %0 : tensor<3x2xf32>
  }
}

