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
      %5 = stablehlo.minimum %arg0, %arg1 : tensor<f16>
      stablehlo.return %5 : tensor<f16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1, 3], scatter_dims_to_operand_dims = [1, 3]>, unique_indices = true} : (tensor<4x2x3x5xf16>, tensor<2xi32>, tensor<4x3xf16>) -> tensor<4x2x3x5xf16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<4x2x3x5xf16>, tensor<4x2x3x5xf16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x2x3x5xf16>, tensor<4x3xf16>) {
    %0 = stablehlo.constant dense<"0xF3BA03BDB548DD425942F3C43ABE6242E93E2BBC54BDD841A641C4C0F0B86A3ED42ACAC0703A1436ECC6F9C46F3E89C2E7BDF3BC82B990BB50C4923F89C01EBCEE45BB4189489F2AA2C2A2301A43C83D91C3C23C1BBAA4330A43C840A5C3823DF3C4D73892C5083C393CB53739BAB43D5336933F6CC11DC40B3D83426BB838C42C41ACC4FBC003C4D23C3ABD653BD23E2D41D941D339F2328CC4634482C07D3B1EB852C2F63D6BC3FE4128B33C3E0DC0FBC35A3AE42E71C06944EC47BC44D6C83B44A53E07BE4D3C14C0A5C5EBC3B8C043BD35BDABC2232E1B4129BF153A94BE3E40DB433FC5023DB3B6F93F6040713E"> : tensor<4x2x3x5xf16>
    %1 = stablehlo.constant dense<[[7.338860e-01, 1.652340e+00, -2.828130e+00], [1.218750e+00, -1.689450e+00, -6.460940e+00], [-1.624020e+00, 3.476560e+00, 7.832030e-01], [-2.097660e+00, -9.106440e-01, -2.068360e+00]]> : tensor<4x3xf16>
    return %0, %1 : tensor<4x2x3x5xf16>, tensor<4x3xf16>
  }
  func.func private @expected() -> tensor<4x2x3x5xf16> {
    %0 = stablehlo.constant dense<"0xF3BA03BDB548DD42DF39F3C43ABE6242E93E2BBC54BDD841A641C4C0A8C16A3ED42ACAC0703A1436ECC6F9C46F3E89C2E7BDF3BC82B990BB50C4923F89C01EBCEE45BB41E03C9F2AA2C2A2301A43C2BE91C3C23C1BBAA43376C6C840A5C3823DF3C4D73892C5083C393CB53739BAB43D5336933F6CC11DC40B3D83426BB838C47FBEACC4FBC003C4D23C3ABD653BD23E2D41D941D339F2328CC4634482C07D3B1EB852C2F63D6BC3FE4128B33C3E0DC0FBC35A3AE42E71C06944EC4732C0D6C83B44A53E07BE49BB14C0A5C5EBC3B8C023C035BDABC2232E1B4129BF153A94BE3E40DB433FC5023DB3B6F93F6040713E"> : tensor<4x2x3x5xf16>
    return %0 : tensor<4x2x3x5xf16>
  }
}

