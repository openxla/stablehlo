// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<32> : tensor<1xi32>
    %1:2 = call @inputs() : () -> (tensor<1x50x3xbf16>, tensor<1x3xbf16>)
    %2 = call @expected() : () -> tensor<1x50x3xbf16>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<bf16>, %arg1: tensor<bf16>):
      %5 = stablehlo.maximum %arg0, %arg1 : tensor<bf16>
      stablehlo.return %5 : tensor<bf16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true} : (tensor<1x50x3xbf16>, tensor<1xi32>, tensor<1x3xbf16>) -> tensor<1x50x3xbf16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<1x50x3xbf16>, tensor<1x50x3xbf16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<1x50x3xbf16>, tensor<1x3xbf16>) {
    %0 = stablehlo.constant dense<"0xFA3FC9BF0CBF63400A404CBE57C0ACBF83C04C4085C006C08A3FDFC05ABEB23FE5BB8740BA3E10C079BB91BF9BBF713FEB40863E1840313F64C0573E80C08CC0BDBCCBBF36C0A0BF14C03940D1C0314065BFF5BF9EBCA7C0C740A2C0733E0A40C240E4BF8C3E86BF5AC14B3FACBF66C02F408640D9BD6240FE4010BF65C0674085BDAEC0A2BF9F3F13C0D6401BC01CC00AC075BF0B3F03BE1640DCBE89C08FBF25BE2ABF86400640824076403E3D11BFB5BF01BF29BF463F50C0C9BFD33FD8BD9A3F97C07B4044BF41C059BF6AC05A4004BD923F14C0F3BCE23EC53E73C0BABFBCBF68C04EBE34C0D4BD1BC000C00A400A410D4029405D4082BD8E3FAAC095C05EBF17C0D33F193FD33F2B3F00C0E2BF26C0814009C0133D114076BFC8BE11C01FBF7FC04ABF284014BFE53F"> : tensor<1x50x3xbf16>
    %1 = stablehlo.constant dense<[[4.750000e+00, 5.390630e-01, -3.554690e-01]]> : tensor<1x3xbf16>
    return %0, %1 : tensor<1x50x3xbf16>, tensor<1x3xbf16>
  }
  func.func private @expected() -> tensor<1x50x3xbf16> {
    %0 = stablehlo.constant dense<"0xFA3FC9BF0CBF63400A404CBE57C0ACBF83C04C4085C006C08A3FDFC05ABEB23FE5BB8740BA3E10C079BB91BF9BBF713FEB40863E1840313F64C0573E80C08CC0BDBCCBBF36C0A0BF14C03940D1C0314065BFF5BF9EBCA7C0C740A2C0733E0A40C240E4BF8C3E86BF5AC14B3FACBF66C02F408640D9BD6240FE4010BF65C0674085BDAEC0A2BF9F3F13C0D6401BC01CC00AC075BF0B3F03BE1640DCBE89C08FBF25BE2ABF86400640824076403E3D11BFB5BF01BF29BF463F50C0C9BFD33FD8BD98400A3F7B4044BF41C059BF6AC05A4004BD923F14C0F3BCE23EC53E73C0BABFBCBF68C04EBE34C0D4BD1BC000C00A400A410D4029405D4082BD8E3FAAC095C05EBF17C0D33F193FD33F2B3F00C0E2BF26C0814009C0133D114076BFC8BE11C01FBF7FC04ABF284014BFE53F"> : tensor<1x50x3xbf16>
    return %0 : tensor<1x50x3xbf16>
  }
}

