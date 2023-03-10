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
      %5 = stablehlo.multiply %arg0, %arg1 : tensor<f16>
      stablehlo.return %5 : tensor<f16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1, 3], scatter_dims_to_operand_dims = [1, 3]>, unique_indices = true} : (tensor<4x2x3x5xf16>, tensor<2xi32>, tensor<4x3xf16>) -> tensor<4x2x3x5xf16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<4x2x3x5xf16>, tensor<4x2x3x5xf16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x2x3x5xf16>, tensor<4x3xf16>) {
    %0 = stablehlo.constant dense<"0x129A65BCF9BFCE3EA3C19444C63D16439FBCE24028C02640593E85265C44383C9D451AC1054541BAE5367B3C2741063F92BD3DBC35B9D43DA6B6F545B63BF82C243BC6C3C9C2F34223C02F450245D8C4303C7243EBC343C35840823B4144AAC1B8C4BBBA3344CEC6544413BAE63C41C2EAC488BC7A414844BFC0504106C1E53D8DBC9A3B5FC4BA34EF3D7F42C03F9BC4F7BE2DC556452FC2C23F2F3BE5452AC4503802C409440745573DA0415D445B35934098BDD4B771BF1AC555C48A435E3AF745E9AE53BA04BD4FBCE33EEB3695BA2041B0C2D33DB74274C5C7C195C0E5BFC63DBF3C9A41303D56BD353B1235812A"> : tensor<4x2x3x5xf16>
    %1 = stablehlo.constant dense<[[-7.753900e-01, 4.667970e+00, -1.780270e+00], [9.467770e-01, -2.302730e+00, 4.832030e+00], [1.878910e+00, -7.832030e-01, 1.111720e+01], [-2.296880e+00, -1.563480e+00, -1.287110e+00]]> : tensor<4x3xf16>
    return %0, %1 : tensor<4x2x3x5xf16>, tensor<4x3xf16>
  }
  func.func private @expected() -> tensor<4x2x3x5xf16> {
    %0 = stablehlo.constant dense<"0x129A65BCF9BFCE3E5F409444C63D16439FBCB34928C02640593E8526C3C7383C9D451AC1054541BAE5367B3C2741063F92BD3DBC35B9D43DA6B6F545B63BF82C243BC6C36DC2F34223C02F4502459449303C7243EBC343C33F49823B4144AAC1B8C4BBBA3344CEC6544413BAE63C41C2EAC488BC7A414844BFC0504106C1E53D46C09A3B5FC4BA34EF3D16C1C03F9BC4F7BE2DC56A532FC2C23F2F3BE5452AC4503802C409440745573DA0415D445B35934098BDD4B771BF1AC555C454C85E3AF745E9AE53BAD83F4FBCE33EEB3695BA99C2B0C2D33DB74274C5C7C195C0E5BFC63DBF3C9A41303D56BD353B1235812A"> : tensor<4x2x3x5xf16>
    return %0 : tensor<4x2x3x5xf16>
  }
}

