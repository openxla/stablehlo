module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<4x4xf32>, tensor<4x1xf32>)
    %1 = call @expected() : () -> tensor<4x1xf32>
    %2 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %3 = stablehlo.constant dense<1> : tensor<i32>
    %4 = stablehlo.constant dense<0> : tensor<i32>
    %5 = stablehlo.constant dense<0> : tensor<i32>
    %6 = stablehlo.constant dense<1> : tensor<i32>
    %7 = stablehlo.constant dense<4> : tensor<i32>
    %8 = stablehlo.constant dense<1> : tensor<i32>
    %9 = stablehlo.constant dense<1> : tensor<i32>
    %10 = stablehlo.custom_call @blas_strsm(%3, %4, %5, %6, %7, %8, %9, %2, %0#0, %0#1) {api_version = 2 : i32, operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 9, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<f32>, tensor<4x4xf32>, tensor<4x1xf32>) -> tensor<4x1xf32>
    %11 = stablehlo.custom_call @check.eq(%10, %1) : (tensor<4x1xf32>, tensor<4x1xf32>) -> tensor<i1>
    return %11 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x4xf32>, tensor<4x1xf32>) {
    %0 = stablehlo.constant dense<[[4.65974617, -3.32539177, 0.678781032, -0.119769022], [1.05173111, -2.3693409, 2.52233124, -2.21890378], [0.815383791, 1.44901788, -2.47812724, -0.392760724], [2.23424149, 4.51241112, 0.437487781, 0.987321197]]> : tensor<4x4xf32>
    %1 = stablehlo.constant dense<[[4.60895252], [-1.5659045], [-0.660932302], [-0.981942415]]> : tensor<4x1xf32>
    return %0, %1 : tensor<4x4xf32>, tensor<4x1xf32>
  }
  func.func private @expected() -> tensor<4x1xf32> {
    %0 = stablehlo.constant dense<[[1.52764559], [-1.10486627], [-1.0466007], [-0.981942415]]> : tensor<4x1xf32>
    return %0 : tensor<4x1xf32>
  }
}
