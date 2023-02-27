module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<4x4xcomplex<f32>>, tensor<4x1xcomplex<f32>>)
    %1 = call @expected() : () -> tensor<4x1xcomplex<f32>>
    %2 = chlo.conj %0#0 : tensor<4x4xcomplex<f32>> -> tensor<4x4xcomplex<f32>>
    %3 = stablehlo.constant dense<(1.000000e+00,0.000000e+00)> : tensor<complex<f32>>
    %4 = stablehlo.constant dense<1> : tensor<i32>
    %5 = stablehlo.constant dense<1> : tensor<i32>
    %6 = stablehlo.constant dense<0> : tensor<i32>
    %7 = stablehlo.constant dense<0> : tensor<i32>
    %8 = stablehlo.constant dense<4> : tensor<i32>
    %9 = stablehlo.constant dense<1> : tensor<i32>
    %10 = stablehlo.constant dense<1> : tensor<i32>
    %11 = stablehlo.custom_call @blas_ctrsm(%4, %5, %6, %7, %8, %9, %10, %3, %2, %0#1) {api_version = 2 : i32, operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 9, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<complex<f32>>, tensor<4x4xcomplex<f32>>, tensor<4x1xcomplex<f32>>) -> tensor<4x1xcomplex<f32>>
    %12 = stablehlo.custom_call @check.eq(%11, %1) : (tensor<4x1xcomplex<f32>>, tensor<4x1xcomplex<f32>>) -> tensor<i1>
    return %12 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x4xcomplex<f32>>, tensor<4x1xcomplex<f32>>) {
    %0 = stablehlo.constant dense<[[(0.0211714152,-6.01056147), (-0.262800336,-1.20185077), (-5.91013622,-0.222525805), (-1.03074217,-3.44610095)], [(5.77146959,-5.453360e-01), (2.89483571,-3.225640e-02), (0.649859071,1.69489324), (-4.75464773,-2.32681155)], [(-5.47544146,2.70615935), (0.270833164,0.948712587), (-0.938555359,0.892912566), (-3.14552021,3.53497386)], [(-1.80108392,-0.771859109), (1.65844214,2.32298851), (-4.52187824,-1.03571856), (-0.695900142,-3.26187134)]]> : tensor<4x4xcomplex<f32>>
    %1 = stablehlo.constant dense<[[(3.74485397,-1.94554234)], [(0.496769369,-2.44768596)], [(-7.7598772,-0.123356275)], [(-4.12797308,4.02819061)]]> : tensor<4x1xcomplex<f32>>
    return %0, %1 : tensor<4x4xcomplex<f32>>, tensor<4x1xcomplex<f32>>
  }
  func.func private @expected() -> tensor<4x1xcomplex<f32>> {
    %0 = stablehlo.constant dense<[[(-0.321488678,-0.624177992)], [(0.700009882,0.451658845)], [(6.78404903,-2.33160543)], [(-5.27553654,-5.27047348)]]> : tensor<4x1xcomplex<f32>>
    return %0 : tensor<4x1xcomplex<f32>>
  }
}
