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
    %0 = stablehlo.constant dense<[[-0.138608485, -0.281959772, 4.72399902, 1.27819633], [-2.30556679, 3.04796815, 0.563946724, -3.602705], [3.564870e-01, -1.57330084, 1.11887264, -2.9574194], [-1.54909623, -0.167918548, -2.45771241, 4.49705553]]> : tensor<4x4xf32>
    %1 = stablehlo.constant dense<[[4.77083874], [-2.94121194], [-0.715548217], [0.322473586]]> : tensor<4x1xf32>
    return %0, %1 : tensor<4x4xf32>, tensor<4x1xf32>
  }
  func.func private @expected() -> tensor<4x1xf32> {
    %0 = stablehlo.constant dense<[[-47.4733658], [-0.796957552], [-0.449987233], [0.0717077181]]> : tensor<4x1xf32>
    return %0 : tensor<4x1xf32>
  }
}
