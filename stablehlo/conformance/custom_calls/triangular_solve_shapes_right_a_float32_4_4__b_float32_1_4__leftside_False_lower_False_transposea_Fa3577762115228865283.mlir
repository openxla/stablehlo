module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<4x4xf32>, tensor<1x4xf32>)
    %1 = call @expected() : () -> tensor<1x4xf32>
    %2 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %3 = stablehlo.constant dense<0> : tensor<i32>
    %4 = stablehlo.constant dense<0> : tensor<i32>
    %5 = stablehlo.constant dense<0> : tensor<i32>
    %6 = stablehlo.constant dense<0> : tensor<i32>
    %7 = stablehlo.constant dense<1> : tensor<i32>
    %8 = stablehlo.constant dense<4> : tensor<i32>
    %9 = stablehlo.constant dense<1> : tensor<i32>
    %10 = stablehlo.custom_call @blas_strsm(%3, %4, %5, %6, %7, %8, %9, %2, %0#0, %0#1) {api_version = 2 : i32, operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 9, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<f32>, tensor<4x4xf32>, tensor<1x4xf32>) -> tensor<1x4xf32>
    %11 = stablehlo.custom_call @check.eq(%10, %1) : (tensor<1x4xf32>, tensor<1x4xf32>) -> tensor<i1>
    return %11 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x4xf32>, tensor<1x4xf32>) {
    %0 = stablehlo.constant dense<[[-0.883202493, -2.73325253, -5.00207186, -0.653419554], [0.563777804, 2.79737377, 3.69755602, 5.22744322], [-1.12512505, -5.25005674, -3.40583301, 3.06945634], [1.29593158, 5.88300705, -2.03769755, 3.19954562]]> : tensor<4x4xf32>
    %1 = stablehlo.constant dense<[[4.66363716, -0.012117587, -1.5972997, -0.668276965]]> : tensor<1x4xf32>
    return %0, %1 : tensor<4x4xf32>, tensor<1x4xf32>
  }
  func.func private @expected() -> tensor<1x4xf32> {
    %0 = stablehlo.constant dense<[[-5.28037119, -5.16366625, 2.61819959, 4.63745499]]> : tensor<1x4xf32>
    return %0 : tensor<1x4xf32>
  }
}
