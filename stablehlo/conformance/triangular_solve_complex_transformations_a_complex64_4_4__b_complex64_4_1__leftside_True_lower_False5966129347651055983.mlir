module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<4x4xcomplex<f32>>, tensor<4x1xcomplex<f32>>)
    %1 = call @expected() : () -> tensor<4x1xcomplex<f32>>
    %2 = stablehlo.constant dense<(1.000000e+00,0.000000e+00)> : tensor<complex<f32>>
    %3 = stablehlo.constant dense<1> : tensor<i32>
    %4 = stablehlo.constant dense<0> : tensor<i32>
    %5 = stablehlo.constant dense<2> : tensor<i32>
    %6 = stablehlo.constant dense<0> : tensor<i32>
    %7 = stablehlo.constant dense<4> : tensor<i32>
    %8 = stablehlo.constant dense<1> : tensor<i32>
    %9 = stablehlo.constant dense<1> : tensor<i32>
    %10 = stablehlo.custom_call @blas_ctrsm(%3, %4, %5, %6, %7, %8, %9, %2, %0#0, %0#1) {api_version = 2 : i32, operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 9, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<complex<f32>>, tensor<4x4xcomplex<f32>>, tensor<4x1xcomplex<f32>>) -> tensor<4x1xcomplex<f32>>
    %11 = stablehlo.custom_call @check.eq(%10, %1) : (tensor<4x1xcomplex<f32>>, tensor<4x1xcomplex<f32>>) -> tensor<i1>
    return %11 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x4xcomplex<f32>>, tensor<4x1xcomplex<f32>>) {
    %0 = stablehlo.constant dense<[[(-1.99633121,-1.62368643), (-0.895778298,4.05792332), (1.00289464,1.31406426), (-3.34500599,-2.298000e+00)], [(1.90690327,-3.23756742), (0.755350172,-0.703075767), (-0.433418155,2.89222121), (1.56829393,-0.646863043)], [(0.727431476,-2.29866552), (0.307433128,2.03629303), (0.735843837,-1.72697771), (-0.0909483805,-1.51186609)], [(-4.03403282,-4.012120e+00), (2.67941499,-3.29511976), (-1.26460731,2.72737813), (-0.0124790436,-2.08149457)]]> : tensor<4x4xcomplex<f32>>
    %1 = stablehlo.constant dense<[[(3.78557825,-2.11824584)], [(5.68124294,0.833371281)], [(-3.11933565,-3.47002077)], [(8.281750e-01,6.59986496)]]> : tensor<4x1xcomplex<f32>>
    return %0, %1 : tensor<4x4xcomplex<f32>>, tensor<4x1xcomplex<f32>>
  }
  func.func private @expected() -> tensor<4x1xcomplex<f32>> {
    %0 = stablehlo.constant dense<[[(-1.66069782,-0.289634526)], [(-0.262074947,-7.91792201)], [(-0.146866322,-12.6357346)], [(10.2242556,13.9767628)]]> : tensor<4x1xcomplex<f32>>
    return %0 : tensor<4x1xcomplex<f32>>
  }
}
