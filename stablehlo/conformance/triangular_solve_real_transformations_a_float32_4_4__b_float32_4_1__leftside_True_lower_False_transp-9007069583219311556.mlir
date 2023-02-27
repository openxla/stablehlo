module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<4x4xf32>, tensor<4x1xf32>)
    %1 = call @expected() : () -> tensor<4x1xf32>
    %2 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %3 = stablehlo.constant dense<1> : tensor<i32>
    %4 = stablehlo.constant dense<0> : tensor<i32>
    %5 = stablehlo.constant dense<1> : tensor<i32>
    %6 = stablehlo.constant dense<0> : tensor<i32>
    %7 = stablehlo.constant dense<4> : tensor<i32>
    %8 = stablehlo.constant dense<1> : tensor<i32>
    %9 = stablehlo.constant dense<1> : tensor<i32>
    %10 = stablehlo.custom_call @blas_strsm(%3, %4, %5, %6, %7, %8, %9, %2, %0#0, %0#1) {api_version = 2 : i32, operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 9, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<f32>, tensor<4x4xf32>, tensor<4x1xf32>) -> tensor<4x1xf32>
    %11 = stablehlo.custom_call @check.eq(%10, %1) : (tensor<4x1xf32>, tensor<4x1xf32>) -> tensor<i1>
    return %11 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x4xf32>, tensor<4x1xf32>) {
    %0 = stablehlo.constant dense<[[0.376304775, -1.62256622, 1.12244904, 2.29549265], [-0.807289302, 4.50351763, 0.149400547, -6.7858429], [1.02234054, 2.00695872, -0.809689581, -3.1098702], [-0.763510525, -0.0488144085, 3.29836464, 1.29760861]]> : tensor<4x4xf32>
    %1 = stablehlo.constant dense<[[-0.791759192], [-4.23274183], [-0.134466916], [3.09548473]]> : tensor<4x1xf32>
    return %0, %1 : tensor<4x4xf32>, tensor<4x1xf32>
  }
  func.func private @expected() -> tensor<4x1xf32> {
    %0 = stablehlo.constant dense<[[-2.10403705], [-1.69793534], [-3.06398916], [-10.1149492]]> : tensor<4x1xf32>
    return %0 : tensor<4x1xf32>
  }
}
