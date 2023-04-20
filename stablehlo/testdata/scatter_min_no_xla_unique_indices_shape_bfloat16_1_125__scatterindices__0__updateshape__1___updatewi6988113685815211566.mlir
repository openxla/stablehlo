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
      %5 = stablehlo.minimum %arg0, %arg1 : tensor<bf16>
      stablehlo.return %5 : tensor<bf16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true} : (tensor<1x125xbf16>, tensor<1xi32>, tensor<1xbf16>) -> tensor<1x125xbf16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<1x125xbf16>, tensor<1x125xbf16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<1x125xbf16>, tensor<1xbf16>) {
    %0 = stablehlo.constant dense<"0x023FAE4056401BBEB93E0DBFACBD104004BFDDBFB3BE034064BF5ABF8140A0C0DC408EC0DBC0C5402E407D408240E1C038C0A3BEA740B0BF2D403FC045C06ABE4D40F13F4F3F02C047C05A40F73F22C05CC02EC01DC015C0864007BE8FBFC93F39C034403A4084BF6F3F41408FC0E23E814016C007C06CBFA34010C053C044C08E3FCABF1C3F22407D4025C039C06A40AE3E903E95C0023F94C0144067C0EDBF2F404540A53FD4BF6E40803F0DC077C06EBF2C3F91BFAAC01E3E2F402EC072BF9A3FC93FE13F953FB73F46BFAA40033F9840753FA9BF9C40ACBFABBF0640A4BF903F12404C406F409CBF1AC08BBF683F58BF61C069402BC0BA3F"> : tensor<1x125xbf16>
    %1 = stablehlo.constant dense<-1.085940e+00> : tensor<1xbf16>
    return %0, %1 : tensor<1x125xbf16>, tensor<1xbf16>
  }
  func.func private @expected() -> tensor<1x125xbf16> {
    %0 = stablehlo.constant dense<"0x8BBFAE4056401BBEB93E0DBFACBD104004BFDDBFB3BE034064BF5ABF8140A0C0DC408EC0DBC0C5402E407D408240E1C038C0A3BEA740B0BF2D403FC045C06ABE4D40F13F4F3F02C047C05A40F73F22C05CC02EC01DC015C0864007BE8FBFC93F39C034403A4084BF6F3F41408FC0E23E814016C007C06CBFA34010C053C044C08E3FCABF1C3F22407D4025C039C06A40AE3E903E95C0023F94C0144067C0EDBF2F404540A53FD4BF6E40803F0DC077C06EBF2C3F91BFAAC01E3E2F402EC072BF9A3FC93FE13F953FB73F46BFAA40033F9840753FA9BF9C40ACBFABBF0640A4BF903F12404C406F409CBF1AC08BBF683F58BF61C069402BC0BA3F"> : tensor<1x125xbf16>
    return %0 : tensor<1x125xbf16>
  }
}

