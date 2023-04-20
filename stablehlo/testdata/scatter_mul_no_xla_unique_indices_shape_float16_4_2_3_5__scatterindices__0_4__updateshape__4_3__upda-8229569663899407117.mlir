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
    %0 = stablehlo.constant dense<"0x80C06E41813ACE3D01C6A4B9A4BFEAB738C356C48A3B52368CC013BCF8AC9F44413C40C3A7483844D934454433BB6B43643D75C66F43E2BD26BEED44B0BD1E451DBF7EC00B3B52B9013072BC2846D4C0433B7AC2EEC0E3C42B43BD3DDCC6343D35C44140BD30054202BDEEBA60BDE7430C31523F17BE80424D40773E25BCDAC2EEC318479EBF4444FDB86E421FB7273E90439C3D2E3E49433EA3C4C1FCBFDAC02444FF3C8EC4A942FFC4743E1C4475C2E940A641E0335C3BA93731C68945C5300CBF16C0EC448240B5B520C45339224138B64634F53F8CC57B45E2C17132882C71C261C13EC1253B6CC833BDCF442CBB"> : tensor<4x2x3x5xf16>
    %1 = stablehlo.constant dense<[[2.781250e+00, 4.218750e+00, -8.359380e-01], [9.956050e-01, -1.187500e+00, 6.806640e-01], [-4.757810e+00, -1.390630e+00, 1.041020e+00], [-2.001950e+00, 4.085940e+00, 4.790040e-01]]> : tensor<4x3xf16>
    return %0, %1 : tensor<4x2x3x5xf16>, tensor<4x3xf16>
  }
  func.func private @expected() -> tensor<4x2x3x5xf16> {
    %0 = stablehlo.constant dense<"0x80C06E41813ACE3D2DCCA4B9A4BFEAB738C393CC8A3B52368CC013BC272C9F44413C40C3A7483844D934454433BB6B43643D75C66F43E2BD26BEED44B0BD1E451DBF7EC0033B52B9013072BC2846BC41433B7AC2EEC0E3C4E140BD3DDCC6343D35C44140BD30054202BDEEBA60BDE7430C31523F17BE80424D40773E25BCDAC2B74C18479EBF4444FDB878C41FB7273E90439C3D6F3E49433EA3C4C1FCBFDAC02444FF3C8EC4A942FFC4743E1C4475C2E940A641E0335C3BA93731C68AC9C5300CBF16C0EC449B48B5B520C453392241F5B14634F53F8CC57B45E2C17132882C71C261C13EC1253B6CC833BDCF442CBB"> : tensor<4x2x3x5xf16>
    return %0 : tensor<4x2x3x5xf16>
  }
}

