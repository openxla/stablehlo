// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<[0, 4]> : tensor<2xi32>
    %1:2 = call @inputs() : () -> (tensor<4x2x3x5xbf16>, tensor<4x3xbf16>)
    %2 = call @expected() : () -> tensor<4x2x3x5xbf16>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<bf16>, %arg1: tensor<bf16>):
      %5 = stablehlo.multiply %arg0, %arg1 : tensor<bf16>
      stablehlo.return %5 : tensor<bf16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1, 3], scatter_dims_to_operand_dims = [1, 3]>, unique_indices = true} : (tensor<4x2x3x5xbf16>, tensor<2xi32>, tensor<4x3xbf16>) -> tensor<4x2x3x5xbf16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<4x2x3x5xbf16>, tensor<4x2x3x5xbf16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x2x3x5xbf16>, tensor<4x3xbf16>) {
    %0 = stablehlo.constant dense<"0xF940B53FA53E353FC03E2C40883FF4BEABC00240FEBFFCBF44C08DBF60BFDFC09F3E3EC0B4BEB5BFA13FF3BF97BFA43F4CC0DCBF8BC06DC0EDBF35403EC0C43FB73F79C017BF963E36C08540813FC9BE3C4011402D40C9BFBF3FCD3FC3BF15C053403B3E34BF944080BE3C3F8EBF144044409D402D3FE03F34C086408B3F8E4063C047C0153F87C023C01A3F224061C07EBF19C006C1AC40DDBF10405BC0AABFF0405BBF83BF49C09E400740F2C018C092BE4CC04CC0C2BF783FAD3FFFBF0040F0BF94C03C406F3E0EC098BE5840D83F16C0A7C0D73EB93F70C0F14038C0AB3E9CC0154002401AC0B23FAB3F27409BC0"> : tensor<4x2x3x5xbf16>
    %1 = stablehlo.constant dense<[[5.156250e+00, 2.359380e+00, 7.656250e-01], [3.921880e+00, -5.781250e-01, 4.531250e+00], [6.500000e+00, -6.750000e+00, -7.382810e-01], [7.656250e-01, 3.000000e+00, 3.875000e+00]]> : tensor<4x3xbf16>
    return %0, %1 : tensor<4x2x3x5xbf16>, tensor<4x3xbf16>
  }
  func.func private @expected() -> tensor<4x2x3x5xbf16> {
    %0 = stablehlo.constant dense<"0xF940B53FA53E353FF83F2C40883FF4BEABC09940FEBFFCBF44C08DBF2CBFDFC09F3E3EC0B4BEB5BFA13FF3BF97BFA43F4CC0DCBF8BC06DC0EDBF35403EC0C43FB73F79C014C0963E36C08540813F683E3C4011402D40C9BFD840CD3FC3BF15C053403B3E34BF944080BE3C3F8EBF144044409D402D3FE03F34C086408B3F8E40B8C147C0153F87C023C082C0224061C07EBF19C0C640AC40DDBF10405BC0AABFF0405BBF83BF49C09E400740F2C018C092BE4CC04CC0C2BF783FAD3FC3BF0040F0BF94C03C40333F0EC098BE5840D83F11C1A7C0D73EB93F70C0F14038C0AB3E9CC0154002401AC0B23FAB3F27409BC0"> : tensor<4x2x3x5xbf16>
    return %0 : tensor<4x2x3x5xbf16>
  }
}

