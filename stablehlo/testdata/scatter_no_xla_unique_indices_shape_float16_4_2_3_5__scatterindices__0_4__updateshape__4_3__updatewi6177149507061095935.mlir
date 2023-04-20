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
    %0 = stablehlo.constant dense<"0x02C5A73A52418538E73CB3C0C2C5FBC4A0C26EB9D942093BFF3EA5423AC409412D28E8C1533F3E40DB465EB9F5C4E0BE2A42E93B10C20BC52FC0DFC1643FEE4394C142293F46E6C2A9C265BE863A313C43C1814261AE8A4065B5A7C4564517C39DC0F02CA7BEF4C18934FE44E9BBB4C1233CCFBA7EBE40C4FABF6D39CDBCC83881C5EDB68EB98AC6AEC5B245DEC49CC16D378FC156BC4BC07344E1454036053DC74365378137CEC338B5B23D32407CC535BE5C454D4159B9B1C4DC3C8AC1EA3C2DC6C135FF3A6E42A5C5F743BEC1D6C614C00CBAB4B58AC0F73F4E40EA3E8445A4424CB69128BAC08E421934E93A32C5"> : tensor<4x2x3x5xf16>
    %1 = stablehlo.constant dense<[[5.609380e+00, -9.680170e-02, 2.515630e+00], [8.484370e+00, 2.271480e+00, -2.534180e-01], [6.567380e-01, 1.240840e-01, -3.273440e+00], [3.294920e+00, 3.833010e-01, 2.787110e+00]]> : tensor<4x3xf16>
    return %0, %1 : tensor<4x2x3x5xf16>, tensor<4x3xf16>
  }
  func.func private @expected() -> tensor<4x2x3x5xf16> {
    %0 = stablehlo.constant dense<"0x02C5A73A524185389C45B3C0C2C5FBC4A0C232AED942093BFF3EA542084109412D28E8C1533F3E40DB465EB9F5C4E0BE2A42E93B10C20BC52FC0DFC1643FEE4394C142293E48E6C2A9C265BE863A8B4043C1814261AE8A400EB4A7C4564517C39DC0F02CA7BEF4C18934FE44E9BBB4C1233CCFBA7EBE40C4FABF6D39CDBCC8384139EDB68EB98AC6AEC5F12FDEC49CC16D378FC18CC24BC07344E1454036053DC74365378137CEC338B5B23D32407CC535BE5C454D4159B9B1C4DC3C9742EA3C2DC6C135FF3A2236A5C5F743BEC1D6C693410CBAB4B58AC0F73F4E40EA3E8445A4424CB69128BAC08E421934E93A32C5"> : tensor<4x2x3x5xf16>
    return %0 : tensor<4x2x3x5xf16>
  }
}

