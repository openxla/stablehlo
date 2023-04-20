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
      stablehlo.return %arg1 : tensor<bf16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1, 3], scatter_dims_to_operand_dims = [1, 3]>, unique_indices = true} : (tensor<4x2x3x5xbf16>, tensor<2xi32>, tensor<4x3xbf16>) -> tensor<4x2x3x5xbf16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<4x2x3x5xbf16>, tensor<4x2x3x5xbf16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x2x3x5xbf16>, tensor<4x3xbf16>) {
    %0 = stablehlo.constant dense<"0x2BBF853F8FBDC6C09F40F4BF84408ABE50BD173EA3C0833F563EDB3F6CC02C3F00BB9D3F2BBF1840C13F6F405A40CFBFCA401040AFC0C8BFABBF27C092C086C05B403DC04F4046C063BF1EC037C015C03A40C83E7DC0284083BF4DBE1BC0424030C088C084BE22BFD23F1B40B940ECBF0340553F1ABFF33F49BEACBF533F80C08BC088401BBFA6C0EA3F2EBF97BF583EF23EC8C016C0A9C0A7BF853FD13FA1BF1BC06340943FD23F1B40A6BE044092BE33C0CC3F0EC099406CC017C09DBF13C030BF9FBE2AC0E1BBA63E8EBFF9C04E3F3FC019C08D3E8AC007C054C0D53DFBC063407C3E413E04BF5FBF95C0F83D4B40"> : tensor<4x2x3x5xbf16>
    %1 = stablehlo.constant dense<[[1.039060e+00, -1.968750e+00, -5.834960e-02], [-5.820310e-01, 4.093750e+00, -4.746090e-01], [-5.195310e-01, 1.660160e-01, -2.968750e+00], [-2.953130e+00, 1.162110e-01, -4.218750e+00]]> : tensor<4x3xbf16>
    return %0, %1 : tensor<4x2x3x5xbf16>, tensor<4x3xbf16>
  }
  func.func private @expected() -> tensor<4x2x3x5xbf16> {
    %0 = stablehlo.constant dense<"0x2BBF853F8FBDC6C0853FF4BF84408ABE50BDFCBFA3C0833F563EDB3F6FBD2C3F00BB9D3F2BBF1840C13F6F405A40CFBFCA401040AFC0C8BFABBF27C092C086C05B403DC015BF46C063BF1EC037C083403A40C83E7DC02840F3BE4DBE1BC0424030C088C084BE22BFD23F1B40B940ECBF0340553F1ABFF33F49BEACBF533F80C005BF88401BBFA6C0EA3F2A3E97BF583EF23EC8C03EC0A9C0A7BF853FD13FA1BF1BC06340943FD23F1B40A6BE044092BE33C0CC3F0EC099406CC017C03DC013C030BF9FBE2AC0EE3DA63E8EBFF9C04E3F87C019C08D3E8AC007C054C0D53DFBC063407C3E413E04BF5FBF95C0F83D4B40"> : tensor<4x2x3x5xbf16>
    return %0 : tensor<4x2x3x5xbf16>
  }
}

