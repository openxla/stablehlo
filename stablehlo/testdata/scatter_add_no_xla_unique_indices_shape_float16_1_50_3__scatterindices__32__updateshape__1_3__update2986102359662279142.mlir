// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = stablehlo.constant dense<32> : tensor<1xi32>
    %1:2 = call @inputs() : () -> (tensor<1x50x3xf16>, tensor<1x3xf16>)
    %2 = call @expected() : () -> tensor<1x50x3xf16>
    %3 = "stablehlo.scatter"(%1#0, %0, %1#1) ({
    ^bb0(%arg0: tensor<f16>, %arg1: tensor<f16>):
      %5 = stablehlo.add %arg0, %arg1 : tensor<f16>
      stablehlo.return %5 : tensor<f16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true} : (tensor<1x50x3xf16>, tensor<1xi32>, tensor<1x3xf16>) -> tensor<1x50x3xf16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<1x50x3xf16>, tensor<1x50x3xf16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<1x50x3xf16>, tensor<1x3xf16>) {
    %0 = stablehlo.constant dense<"0xD8406C47DF40B0BF4CC69AA66D4382C40432FA461AB26832294047BC573BC6C05BB0E3BA1D46594135C44C3FF9C167BDB5B47BB8B4B96E48D9BEDA371E3DB74365C291A941BEA7C28F4218C526301C460BC27A44E831452E89C094C70038B9421BC629BD29C2EABAE14298C2753D994471BFE23C47C1D045E32FA84301C2B33FE4402FC050BCA43B323EDF440B452C43BC45934571C6EB3F4F4342B8AE3A014218C1C2C3CBBA2B3DE0BE9A2E01B42E4092BDB5C437BDF7C2BC3DDA40D8C56E40D43B00C1CE416B4394BE6C4192440940CFB3EDC4813E3239883E763E0C3190C41C40354036385740BCC272C706C18939BF3D75C15E4225409BC3074489C7F1438BC511BC88BCDFBB024439C23743743E90BFFCBF373849409DC020458945F2C002C42639B638A4C463B499BB"> : tensor<1x50x3xf16>
    %1 = stablehlo.constant dense<[[-4.038090e-01, -1.107420e+00, -3.810550e+00]]> : tensor<1x3xf16>
    return %0, %1 : tensor<1x50x3xf16>, tensor<1x3xf16>
  }
  func.func private @expected() -> tensor<1x50x3xf16> {
    %0 = stablehlo.constant dense<"0xD8406C47DF40B0BF4CC69AA66D4382C40432FA461AB26832294047BC573BC6C05BB0E3BA1D46594135C44C3FF9C167BDB5B47BB8B4B96E48D9BEDA371E3DB74365C291A941BEA7C28F4218C526301C460BC27A44E831452E89C094C70038B9421BC629BD29C2EABAE14298C2753D994471BFE23C47C1D045E32FA84301C2B33FE4402FC050BCA43B323EDF440B452C43BC45934571C6EB3F4F4342B8AE3A014218C1C2C3CBBA2B3DE0BE9A2E01B42E4092BDB5C437BDF7C2BC3DDA40D8C56E40993837C344BB6B4394BE6C4192440940CFB3EDC4813E3239883E763E0C3190C41C40354036385740BCC272C706C18939BF3D75C15E4225409BC3074489C7F1438BC511BC88BCDFBB024439C23743743E90BFFCBF373849409DC020458945F2C002C42639B638A4C463B499BB"> : tensor<1x50x3xf16>
    return %0 : tensor<1x50x3xf16>
  }
}

