module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<4x4xcomplex<f32>>
    %1 = call @expected() : () -> tensor<4x4xcomplex<f32>>
    %2 = stablehlo.constant dense<1> : tensor<i32>
    %3 = stablehlo.constant dense<1> : tensor<i32>
    %4 = stablehlo.constant dense<4> : tensor<i32>
    %5 = stablehlo.custom_call @lapack_cpotrf(%2, %3, %4, %0) {api_version = 2 : i32, operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 3, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<> : tensor<0xindex>]} : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<4x4xcomplex<f32>>) -> tuple<tensor<4x4xcomplex<f32>>, tensor<i32>>
    %6 = stablehlo.get_tuple_element %5[0] : (tuple<tensor<4x4xcomplex<f32>>, tensor<i32>>) -> tensor<4x4xcomplex<f32>>
    %7 = stablehlo.get_tuple_element %5[1] : (tuple<tensor<4x4xcomplex<f32>>, tensor<i32>>) -> tensor<i32>
    %8 = stablehlo.constant dense<0> : tensor<i32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<i32>) -> tensor<i32>
    %10 = stablehlo.compare  EQ, %7, %9,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %11 = stablehlo.broadcast_in_dim %10, dims = [] : (tensor<i1>) -> tensor<1x1xi1>
    %12 = stablehlo.constant dense<(0x7FC00000,0x7FC00000)> : tensor<complex<f32>>
    %13 = stablehlo.broadcast_in_dim %12, dims = [] : (tensor<complex<f32>>) -> tensor<4x4xcomplex<f32>>
    %14 = stablehlo.broadcast_in_dim %11, dims = [0, 1] : (tensor<1x1xi1>) -> tensor<4x4xi1>
    %15 = stablehlo.select %14, %6, %13 : tensor<4x4xi1>, tensor<4x4xcomplex<f32>>
    %16 = stablehlo.custom_call @check.eq(%15, %1) : (tensor<4x4xcomplex<f32>>, tensor<4x4xcomplex<f32>>) -> tensor<i1>
    return %16 : tensor<i1>
  }
  func.func private @inputs() -> tensor<4x4xcomplex<f32>> {
    %0 = stablehlo.constant dense<[[(96.3251495,0.000000e+00), (-57.1368675,-42.6700439), (-34.7039948,-13.9380331), (-9.14440727,2.64007664)], [(-57.1368675,42.6700439), (124.03093,0.000000e+00), (37.8510475,-8.77538871), (11.9887714,-4.36750078)], [(-34.7039948,13.9380331), (37.8510475,8.77538871), (7.578400e+01,0.000000e+00), (-4.89551353,45.4381638)], [(-9.14440727,-2.64007664), (11.9887714,4.36750078), (-4.89551353,-45.4381638), (40.9580269,0.000000e+00)]]> : tensor<4x4xcomplex<f32>>
    return %0 : tensor<4x4xcomplex<f32>>
  }
  func.func private @expected() -> tensor<4x4xcomplex<f32>> {
    %0 = stablehlo.constant dense<[[(9.814538,0.000000e+00), (-57.1368675,-42.6700439), (-34.7039948,-13.9380331), (-9.14440727,2.64007664)], [(-5.82165623,4.34763622), (8.440220e+00,0.000000e+00), (37.8510475,-8.77538871), (11.9887714,-4.36750078)], [(-3.53597832,1.42014146), (1.31412899,0.197840452), (7.71349334,0.000000e+00), (-4.89551353,45.4381638)], [(-0.931720554,-0.268996507), (0.916340291,-0.148015887), (-1.16457617,-6.136870e+00), (0.372243285,0.000000e+00)]]> : tensor<4x4xcomplex<f32>>
    return %0 : tensor<4x4xcomplex<f32>>
  }
}
