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
      %5 = stablehlo.multiply %arg0, %arg1 : tensor<f16>
      stablehlo.return %5 : tensor<f16>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true} : (tensor<1x50x3xf16>, tensor<1xi32>, tensor<1x3xf16>) -> tensor<1x50x3xf16>
    %4 = stablehlo.custom_call @check.eq(%3, %2) : (tensor<1x50x3xf16>, tensor<1x50x3xf16>) -> tensor<i1>
    return %4 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<1x50x3xf16>, tensor<1x3xf16>) {
    %0 = stablehlo.constant dense<"0x9FBE05C3CCBF40C450C1483CE344D9BCF23C183EF640EB3AECAC8242B844793F164229B151C161BE16446C46F038EAC2B3AC7A4451BEE33CD1B430C502BC7344853B0B411F4319433C37A53551C43144634098BF8CB8FEB020C23E3AB3C2443B22C5D43D56AFC13B32BD613CDBBCD8AD97407CBC373883BCCBC40EC2202B7FBFDCC1E2BF71302A42C1B36FC09A4336440A41E744EA2A82C4FF3A10B7E64124434EB721C11A3F9DBAE4BF8FC4FE3B364595442DB669C4AC405DBB5C4188B9B43A0932A2C1933BA8407D27393CD7382AC20C4209C2F2BC3940A735883F31BC8FBF1CADE5428444DC3FF6C4CF3716C254407FC57F44253E74C12B423D41B744B7B49C408C445238E8B893474AC115C252B7CBB8F0C13BC23A3CB1C570B600B5E1C06CB64BBF2B3EE845C64659BF"> : tensor<1x50x3xf16>
    %1 = stablehlo.constant dense<[[6.207030e+00, 4.750000e+00, -8.007810e-01]]> : tensor<1x3xf16>
    return %0, %1 : tensor<1x50x3xf16>, tensor<1x3xf16>
  }
  func.func private @expected() -> tensor<1x50x3xf16> {
    %0 = stablehlo.constant dense<"0x9FBE05C3CCBF40C450C1483CE344D9BCF23C183EF640EB3AECAC8242B844793F164229B151C161BE16446C46F038EAC2B3AC7A4451BEE33CD1B430C502BC7344853B0B411F4319433C37A53551C43144634098BF8CB8FEB020C23E3AB3C2443B22C5D43D56AFC13B32BD613CDBBCD8AD97407CBC373883BCCBC40EC2202B7FBFDCC1E2BF71302A42C1B36FC09A4336440A41E744EA2A82C4FF3A10B7E64124434EB721C11A3F9DBAE4BF8FC4FE3B364595442DB669C4AC405DBB5C4188B9B43AAF3CB0CA11BAA8407D27393CD7382AC20C4209C2F2BC3940A735883F31BC8FBF1CADE5428444DC3FF6C4CF3716C254407FC57F44253E74C12B423D41B744B7B49C408C445238E8B893474AC115C252B7CBB8F0C13BC23A3CB1C570B600B5E1C06CB64BBF2B3EE845C64659BF"> : tensor<1x50x3xf16>
    return %0 : tensor<1x50x3xf16>
  }
}

