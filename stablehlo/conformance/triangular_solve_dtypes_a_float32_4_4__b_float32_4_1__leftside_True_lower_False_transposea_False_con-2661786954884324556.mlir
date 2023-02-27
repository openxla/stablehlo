module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<4x4xf32>, tensor<4x1xf32>)
    %1 = call @expected() : () -> tensor<4x1xf32>
    %2 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %3 = stablehlo.constant dense<1> : tensor<i32>
    %4 = stablehlo.constant dense<0> : tensor<i32>
    %5 = stablehlo.constant dense<0> : tensor<i32>
    %6 = stablehlo.constant dense<0> : tensor<i32>
    %7 = stablehlo.constant dense<4> : tensor<i32>
    %8 = stablehlo.constant dense<1> : tensor<i32>
    %9 = stablehlo.constant dense<1> : tensor<i32>
    %10 = stablehlo.custom_call @blas_strsm(%3, %4, %5, %6, %7, %8, %9, %2, %0#0, %0#1) {api_version = 2 : i32, operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 9, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<f32>, tensor<4x4xf32>, tensor<4x1xf32>) -> tensor<4x1xf32>
    %11 = stablehlo.custom_call @check.eq(%10, %1) : (tensor<4x1xf32>, tensor<4x1xf32>) -> tensor<i1>
    return %11 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x4xf32>, tensor<4x1xf32>) {
    %0 = stablehlo.constant dense<[[-2.81768179, 5.89424133, -3.5455718, -0.632472038], [5.02953815, -0.952798783, 2.0136373, 4.96875668], [-1.70915496, -1.23470938, -1.31260192, 4.54779339], [-5.58991957, 2.90960813, 3.28678083, 0.396593183]]> : tensor<4x4xf32>
    %1 = stablehlo.constant dense<[[0.541502535], [3.55775476], [-1.94938672], [-4.47537184]]> : tensor<4x1xf32>
    return %0, %1 : tensor<4x4xf32>, tensor<4x1xf32>
  }
  func.func private @expected() -> tensor<4x1xf32> {
    %0 = stablehlo.constant dense<[[-247.527176], [-1.420720e+02], [-37.6126022], [-11.2845411]]> : tensor<4x1xf32>
    return %0 : tensor<4x1xf32>
  }
}
