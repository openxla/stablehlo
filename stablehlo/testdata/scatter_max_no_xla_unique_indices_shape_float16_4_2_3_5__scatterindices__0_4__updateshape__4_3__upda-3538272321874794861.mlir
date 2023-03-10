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
      %5 = stablehlo.maximum %arg0, %arg1 : tensor<f16>
      stablehlo.return %5 : tensor<f16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1, 3], scatter_dims_to_operand_dims = [1, 3]>, unique_indices = true} : (tensor<4x2x3x5xf16>, tensor<2xi32>, tensor<4x3xf16>) -> tensor<4x2x3x5xf16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<4x2x3x5xf16>, tensor<4x2x3x5xf16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x2x3x5xf16>, tensor<4x3xf16>) {
    %0 = stablehlo.constant dense<"0xE344823C2E419C3B00C12D3BF9C140BEBF428A419BBAEA40AF40B32A9AAA97AB42C134BD1940504098C40542F4432B344043AB4458BBC7BFA03A833CF4BFF2420FBEE430933933419E3F513AF8BDF832D34001BB3AC0583B9E3DA2C337C09F41A0C5C0BE6FB6FAC0BE38CEBE263E54C2333EF3BDB245DBC3F7B90DC440B6A3BE4C4301BEBF3DAE44DEBC67417C4159415BC21544AE332C418A4004BE6D39994072C0C8C2F04473C12F3C1CBC10B4213AA442ABC5343C1FA83EC1C33EC6390641B3422BC23432E2414042A133C744BAB4F23C7EBF85BD5FC1983C3D426D3E65C194C38EC03AC436BF98BD9BB6824303BF"> : tensor<4x2x3x5xf16>
    %1 = stablehlo.constant dense<[[-1.767580e+00, -4.308590e+00, 5.035160e+00], [-1.650390e+00, 4.089840e+00, 2.683590e+00], [-1.509770e+00, 4.355470e+00, 1.296390e-01], [3.315430e-01, 6.064450e-01, 9.609370e-01]]> : tensor<4x3xf16>
    return %0, %1 : tensor<4x2x3x5xf16>, tensor<4x3xf16>
  }
  func.func private @expected() -> tensor<4x2x3x5xf16> {
    %0 = stablehlo.constant dense<"0xE344823C2E419C3B12BF2D3BF9C140BEBF428A419BBAEA40AF40B32A094597AB42C134BD1940504098C40542F4432B344043AB4458BBC7BFA03A833CF4BFF2420FBEE430933933419E3F513AF8BD1744D34001BB3AC0583B5E41A2C337C09F41A0C5C0BE6FB6FAC0BE38CEBE263E54C2333EF3BDB245DBC3F7B90DC440B6A3BE4C4301BEBF3DAE44DEBC5B447C4159415BC21544AE332C418A4004BE6D39994072C0C8C2F04473C12F3C1CBC10B4213AA442ABC5343C1FA83EC1C33EC6390641B3422BC23432E2414042A133C744BAB4F23C7EBF85BD5FC1983C3D426D3E65C194C38EC03AC436BF98BD9BB6824303BF"> : tensor<4x2x3x5xf16>
    return %0 : tensor<4x2x3x5xf16>
  }
}

