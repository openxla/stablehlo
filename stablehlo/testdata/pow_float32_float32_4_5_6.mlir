// RUN: stablehlo-opt -inline %s | stablehlo-translate --interpret
// RUN: stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt > %t.0
// RUN: stablehlo-opt %s > %t.1
// RUN: diff %t.0 %t.1

module @jit_main attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main() -> (tensor<4x5x6xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0:2 = call @inputs() : () -> (tensor<f32>, tensor<4x5x6xf32>)
    %1 = call @expected() : () -> tensor<4x5x6xf32>
    %2 = stablehlo.broadcast_in_dim %0#0, dims = [] : (tensor<f32>) -> tensor<4x5x6xf32>
    %3 = stablehlo.power %2, %0#1 : tensor<4x5x6xf32>
    stablehlo.custom_call @check.expect_close(%3, %1) {has_side_effect = true} : (tensor<4x5x6xf32>, tensor<4x5x6xf32>) -> ()
    return %3 : tensor<4x5x6xf32>
  }
  func.func private @inputs() -> (tensor<f32> {mhlo.layout_mode = "default"}, tensor<4x5x6xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<"0x7CE558C029AFCAC01AF2D63F1568DDBF92C768BF57FCF1C079BBB8BF2CB3CBBD1E9377BF2889CA3F1C12823F9A68CFBF4E220DC07D5D0BC1C98781C04926384022A93DC05F7A0141185A1640A2EA3640AB68263E4298D0BF5B95D0BF1A0507C068D6CF3FF20246C018ECA6BFC08F09BF32E96240B46DF13F4122D4C089DE3B40D0563B400EF9D53F1783F3BFE00DFFC05FBC8D400F41C9C0B992174004C2C9BF79F36FC011B4CF3E19000B407188863F5AC6B53C174544BF647490C0951D41C0B4EB98C053E64FC004EA0340AA5314409DF09EBE9614A5C08A29C83E2C0F6EC03FF265406FF1C1BF9FAC31C0DAE3E4BF510618BF0AFD9740296A9C3FF715BABF99346040235AFE3F57E25140BC8001BF46084B4081327DC078254BC0CDD0E83FA06478C0E25DBD402B118B40693628BEF591A0BF8415ADBF9D84DEC0901211BFC47C09C02E0D9FC0DD537A407194034037009A3F0D8AD63F80DB6BC0F7C887C04196DEBFE6F7173F2E03A93FEFC6F9C0C3A21BC0204BE5BE670B4D4026473A40A18B8B40E83C94C08C7B91C097A5B8BF5E38F13D751867BF56ED96BF20E1273FFCF191BFA71AC8BFD8C5A6BFBC7407400378CABF15611BC096270C3F0A37063FCF6E65404D6483C0AEF4E23EB9381EC086A512C0528EAD40A090A73F37ECAABF"> : tensor<4x5x6xf32>
    %cst_0 = stablehlo.constant dense<-1.75889099> : tensor<f32>
    return %cst_0, %cst : tensor<f32>, tensor<4x5x6xf32>
  }
  func.func private @expected() -> (tensor<4x5x6xf32> {mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<0xFFC00000> : tensor<4x5x6xf32>
    return %cst : tensor<4x5x6xf32>
  }
}
