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
      stablehlo.return %arg1 : tensor<f16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1, 3], scatter_dims_to_operand_dims = [1, 3]>, unique_indices = true} : (tensor<4x2x3x5xf16>, tensor<2xi32>, tensor<4x3xf16>) -> tensor<4x2x3x5xf16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<4x2x3x5xf16>, tensor<4x2x3x5xf16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x2x3x5xf16>, tensor<4x3xf16>) {
    %0 = stablehlo.constant dense<"0x6ABE923DDB46CC4140C1AD45213D4CBD773F8B41683A54BF3BBA7E45A43BC1C4A8417CB414C4E6C085BDB2C5FC3CEABF544526BC264478C06F4070B90739F54556C0B54176C4283F31B58340B6B68ABF5A43BE3D6C42304185404CBED733FB446647BBC029BF953C9C449DC446C111B70B44E12FDE3C504600BF0FBD08432DBCC73D573C774453422144C1387743BE42473FEEC038C28A2F134428BD40C59B3DD3C52940BCBA15B5E3BA2BC0813A8D40C6ABCDC73B4679472B411C436942A53E0537D3BE3136DCC106C063C57544494448BC34C877C562B6B3C4672C04C427BE0046E5403BBA044151B79CC443B54DC4"> : tensor<4x2x3x5xf16>
    %1 = stablehlo.constant dense<[[6.207030e+00, 7.827140e-01, 2.595700e+00], [-4.289060e+00, 1.799800e+00, -1.779300e+00], [5.014650e-01, -2.375000e+00, -2.925780e+00], [-2.906250e+00, 2.482420e+00, 3.593750e+00]]> : tensor<4x3xf16>
    return %0, %1 : tensor<4x2x3x5xf16>, tensor<4x3xf16>
  }
  func.func private @expected() -> tensor<4x2x3x5xf16> {
    %0 = stablehlo.constant dense<"0x6ABE923DDB46CC413546AD45213D4CBD773F433A683A54BF3BBA7E453141C1C4A8417CB414C4E6C085BDB2C5FC3CEABF544526BC264478C06F4070B90739F54556C0B5414AC4283F31B58340B6B6333F5A43BE3D6C4230411EBF4CBED733FB446647BBC029BF953C9C449DC446C111B70B44E12FDE3C504600BF0FBD08432DBC0338573C774453422144C0C07743BE42473FEEC0DAC18A2F134428BD40C59B3DD3C52940BCBA15B5E3BA2BC0813A8D40C6ABCDC73B4679472B411C43D0C1A53E0537D3BE3136F74006C063C575444944304334C877C562B6B3C4672C04C427BE0046E5403BBA044151B79CC443B54DC4"> : tensor<4x2x3x5xf16>
    return %0 : tensor<4x2x3x5xf16>
  }
}

