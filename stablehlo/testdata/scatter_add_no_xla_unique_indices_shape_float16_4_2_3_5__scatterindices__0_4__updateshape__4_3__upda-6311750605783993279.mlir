// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<[0, 4]> : tensor<2xi32>
    %1:2 = call @inputs() : () -> (tensor<4x2x3x5xf16>, tensor<4x3xf16>)
    %2 = call @expected() : () -> tensor<4x2x3x5xf16>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<f16>, %arg1: tensor<f16>):
      %5 = stablehlo.add %arg0, %arg1 : tensor<f16>
      stablehlo.return %5 : tensor<f16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1, 3], scatter_dims_to_operand_dims = [1, 3]>, unique_indices = true} : (tensor<4x2x3x5xf16>, tensor<2xi32>, tensor<4x3xf16>) -> tensor<4x2x3x5xf16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<4x2x3x5xf16>, tensor<4x2x3x5xf16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x2x3x5xf16>, tensor<4x3xf16>) {
    %0 = stablehlo.constant dense<"0x95BCD63EB6C33AC4DFBBE1C06FBEB9C43B41873FE13CB93C1A40C33131A7E64521BC9945DA3818444ABD14437BBDD940D7C252C0EA3DBAC7AFBF95C576C23E3D7B3C1B4009BEECBF1A2B14C50CBA904477C5AEC409C2A03F8EBE10BC233A573473C5CAC10842371D543EC14259B633C5C743F2BC873DFFC032BD9A4079C1BABE3FC2C93D1A4221C47BC72FBDA444974451C175B1D73C7EC02140BBAF2143163F74BE09B2A5404342953F2AC71F38CEC210C5B03D41C16C45A636C441EEC23CB849BAB145A53C24385F42B53DC73C4F3E0BC0E4C19EC13FB9A5BD5DC4C53D96BEBBB848ADD845773403A928C64AC80CB8"> : tensor<4x2x3x5xf16>
    %1 = stablehlo.constant dense<[[-5.289060e+00, -3.974610e-01, 3.148440e+00], [2.166020e+00, 7.843020e-02, -2.750000e+00], [1.483400e+00, 1.851560e+00, 6.708980e-01], [1.049800e+00, 2.169920e+00, -2.234380e+00]]> : tensor<4x3xf16>
    return %0, %1 : tensor<4x2x3x5xf16>, tensor<4x3xf16>
  }
  func.func private @expected() -> tensor<4x2x3x5xf16> {
    %0 = stablehlo.constant dense<"0x95BCD63EB6C33AC446C6E1C06FBEB9C43B41F03DE13CB93C1A40C3313E42E64521BC9945DA3818444ABD14437BBDD940D7C252C0EA3DBAC7AFBF95C576C23E3D7B3C1B404239ECBF1A2B14C50CBAA44477C5AEC409C2A03F64C410BC233A573473C5CAC10842371D543EC14259B633C5C743F2BC873DFFC032BD9A4079C1BABE8FBEC93D1A4221C47BC77238A444974451C175B1863F7EC02140BBAF2143163F74BE09B2A5404342953F2AC71F38CEC210C5B03D41C16C45A636C441D4C03CB849BAB145A53C60415F42B53DC73C4F3E42C4E4C19EC13FB9A5BD5DC4C53D96BEBBB848ADD845773403A928C64AC80CB8"> : tensor<4x2x3x5xf16>
    return %0 : tensor<4x2x3x5xf16>
  }
}

