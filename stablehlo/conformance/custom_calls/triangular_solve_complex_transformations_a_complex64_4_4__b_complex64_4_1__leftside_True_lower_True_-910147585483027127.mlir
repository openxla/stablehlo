module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<4x4xcomplex<f32>>, tensor<4x1xcomplex<f32>>)
    %1 = call @expected() : () -> tensor<4x1xcomplex<f32>>
    %2 = stablehlo.constant dense<(1.000000e+00,0.000000e+00)> : tensor<complex<f32>>
    %3 = stablehlo.constant dense<1> : tensor<i32>
    %4 = stablehlo.constant dense<1> : tensor<i32>
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
    %0 = stablehlo.constant dense<[[(0.489298761,6.93502188), (-0.3509821,6.00016975), (0.957040846,2.78282046), (3.14070845,1.82182729)], [(0.762153446,0.742374479), (-3.32270336,-3.67474294), (-0.104961246,5.24128675), (-1.95542085,-2.29475188)], [(-4.379860e-01,-2.56962013), (1.0548842,-2.01543617), (0.732898294,0.629795134), (-0.212370038,-6.70596075)], [(4.903360e+00,5.69436598), (1.55524671,-4.531450e+00), (-1.32033253,-3.52082825), (2.71538448,0.47145468)]]> : tensor<4x4xcomplex<f32>>
    %1 = stablehlo.constant dense<[[(-2.00721169,3.78422427)], [(0.310787112,-5.89499426)], [(3.23989582,-0.845757365)], [(0.0186085142,-3.79696178)]]> : tensor<4x1xcomplex<f32>>
    return %0, %1 : tensor<4x4xcomplex<f32>>, tensor<4x1xcomplex<f32>>
  }
  func.func private @expected() -> tensor<4x1xcomplex<f32>> {
    %0 = stablehlo.constant dense<[[(-0.877104401,-0.966914832)], [(1.42019725,2.75980902)], [(1.39980483,-3.5585463)], [(0.242328316,-1.35624063)]]> : tensor<4x1xcomplex<f32>>
    return %0 : tensor<4x1xcomplex<f32>>
  }
}
