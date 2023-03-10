// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<0> : tensor<1xi32>
    %1:2 = call @inputs() : () -> (tensor<1x125xf16>, tensor<1xf16>)
    %2 = call @expected() : () -> tensor<1x125xf16>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<f16>, %arg1: tensor<f16>):
      %5 = stablehlo.add %arg0, %arg1 : tensor<f16>
      stablehlo.return %5 : tensor<f16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true} : (tensor<1x125xf16>, tensor<1xi32>, tensor<1xf16>) -> tensor<1x125xf16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<1x125xf16>, tensor<1x125xf16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<1x125xf16>, tensor<1xf16>) {
    %0 = stablehlo.constant dense<"0x81443C4264390B4046B56C3F37B7BCC137B62B44E2C1914233C24D3F2CBCBC400EBCEFBBCD412FA71F3B2EC419B99AB4D7C0B1C0A5355043B64083C2EF43A6BC05BBD7C0B4BF1AB8EDBF1345AA449AC43AC303C2A0BCC8428E4121401A4182BA0BC5622A00C271C4BEB34DBD9F3555C21EBC60C4F6C0A6C3FC3D4FBC2E3402C2263D80B4CE402E4400B853C07FAC81C19CC148431C3ED5BBF03E9836A6C4D7380D3F83BF36C440C123C06E40EEC2BC45EFC176C4F6B1CEC3CCC22D3C243CB03CA7BED0B5983DD1439E3764C1B7B3854001BF7F4445BC3F2B08C1B33F50C0583A173FB333284509C205C594C322C4763EDD3CFD384DC113C194C1"> : tensor<1x125xf16>
    %1 = stablehlo.constant dense<2.746090e+00> : tensor<1xf16>
    return %0, %1 : tensor<1x125xf16>, tensor<1xf16>
  }
  func.func private @expected() -> tensor<1x125xf16> {
    %0 = stablehlo.constant dense<"0x40473C4264390B4046B56C3F37B7BCC137B62B44E2C1914233C24D3F2CBCBC400EBCEFBBCD412FA71F3B2EC419B99AB4D7C0B1C0A5355043B64083C2EF43A6BC05BBD7C0B4BF1AB8EDBF1345AA449AC43AC303C2A0BCC8428E4121401A4182BA0BC5622A00C271C4BEB34DBD9F3555C21EBC60C4F6C0A6C3FC3D4FBC2E3402C2263D80B4CE402E4400B853C07FAC81C19CC148431C3ED5BBF03E9836A6C4D7380D3F83BF36C440C123C06E40EEC2BC45EFC176C4F6B1CEC3CCC22D3C243CB03CA7BED0B5983DD1439E3764C1B7B3854001BF7F4445BC3F2B08C1B33F50C0583A173FB333284509C205C594C322C4763EDD3CFD384DC113C194C1"> : tensor<1x125xf16>
    return %0 : tensor<1x125xf16>
  }
}

