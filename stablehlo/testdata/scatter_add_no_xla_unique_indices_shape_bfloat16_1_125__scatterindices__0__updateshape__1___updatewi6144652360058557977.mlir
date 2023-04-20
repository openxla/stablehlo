// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<0> : tensor<1xi32>
    %1:2 = call @inputs() : () -> (tensor<1x125xbf16>, tensor<1xbf16>)
    %2 = call @expected() : () -> tensor<1x125xbf16>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<bf16>, %arg1: tensor<bf16>):
      %5 = stablehlo.add %arg0, %arg1 : tensor<bf16>
      stablehlo.return %5 : tensor<bf16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true} : (tensor<1x125xbf16>, tensor<1xi32>, tensor<1xbf16>) -> tensor<1x125xbf16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<1x125xbf16>, tensor<1x125xbf16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<1x125xbf16>, tensor<1xbf16>) {
    %0 = stablehlo.constant dense<"0xA93FCE3F16C077408D408E405D3F844066C01BC0E03F13C006C023C0A8C07CC0CD3FBD3FC63FAE3F23C04BC0E140D3C051C034C0C840D8BFDEBF673E95407CBE8E3F7F40AFC09A3F98C0AB3FB13EE3BE7F3F9CBF0AC00BC0F3BF1EC05E407CBF9EC094BFA1C042BF193E57C00C3F2D3FC23F04C0CCBF61C092C0F3BF66BE7F3EC53F2740974042C0A5BFE9BEC04096C094BF833F9CC0574096401140623F9D3FEFBF6C3F0EC09A40E43F03C010C0173F67C0D0BD94BFE8BF38C06640E83E45C08FC09C3EFDBF82C0D93E9EBF37407CBE40408840244013408240EDBF16C0C9BF5CC0913FD2C0703FF73F904098BFC8C0CBBF1CC01B40A33F32BF"> : tensor<1x125xbf16>
    %1 = stablehlo.constant dense<1.320310e+00> : tensor<1xbf16>
    return %0, %1 : tensor<1x125xbf16>, tensor<1xbf16>
  }
  func.func private @expected() -> tensor<1x125xbf16> {
    %0 = stablehlo.constant dense<"0x2940CE3F16C077408D408E405D3F844066C01BC0E03F13C006C023C0A8C07CC0CD3FBD3FC63FAE3F23C04BC0E140D3C051C034C0C840D8BFDEBF673E95407CBE8E3F7F40AFC09A3F98C0AB3FB13EE3BE7F3F9CBF0AC00BC0F3BF1EC05E407CBF9EC094BFA1C042BF193E57C00C3F2D3FC23F04C0CCBF61C092C0F3BF66BE7F3EC53F2740974042C0A5BFE9BEC04096C094BF833F9CC0574096401140623F9D3FEFBF6C3F0EC09A40E43F03C010C0173F67C0D0BD94BFE8BF38C06640E83E45C08FC09C3EFDBF82C0D93E9EBF37407CBE40408840244013408240EDBF16C0C9BF5CC0913FD2C0703FF73F904098BFC8C0CBBF1CC01B40A33F32BF"> : tensor<1x125xbf16>
    return %0 : tensor<1x125xbf16>
  }
}

