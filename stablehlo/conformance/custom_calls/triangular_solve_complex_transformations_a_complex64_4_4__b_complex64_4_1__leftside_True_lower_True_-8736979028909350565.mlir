module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<4x4xcomplex<f32>>, tensor<4x1xcomplex<f32>>)
    %1 = call @expected() : () -> tensor<4x1xcomplex<f32>>
    %2 = stablehlo.constant dense<(1.000000e+00,0.000000e+00)> : tensor<complex<f32>>
    %3 = stablehlo.constant dense<1> : tensor<i32>
    %4 = stablehlo.constant dense<1> : tensor<i32>
    %5 = stablehlo.constant dense<0> : tensor<i32>
    %6 = stablehlo.constant dense<0> : tensor<i32>
    %7 = stablehlo.constant dense<4> : tensor<i32>
    %8 = stablehlo.constant dense<1> : tensor<i32>
    %9 = stablehlo.constant dense<1> : tensor<i32>
    %10 = stablehlo.custom_call @blas_ctrsm(%3, %4, %5, %6, %7, %8, %9, %2, %0#0, %0#1) {api_version = 2 : i32, operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>, dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 9, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>]} : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<complex<f32>>, tensor<4x4xcomplex<f32>>, tensor<4x1xcomplex<f32>>) -> tensor<4x1xcomplex<f32>>
    %11 = stablehlo.custom_call @check.eq(%10, %1) : (tensor<4x1xcomplex<f32>>, tensor<4x1xcomplex<f32>>) -> tensor<i1>
    return %11 : tensor<i1>
  }
  func.func private @inputs() -> (tensor<4x4xcomplex<f32>>, tensor<4x1xcomplex<f32>>) {
    %0 = stablehlo.constant dense<[[(-3.23808217,-1.0020442), (-0.0558407493,5.12425566), (7.46130561,6.27637386), (0.263719231,0.14313291)], [(2.69195509,-6.3085947), (-2.35308909,3.57929754), (3.06726813,-1.20254397), (-0.251245439,7.96047592)], [(0.162099108,3.47823882), (-0.545323312,-5.12168741), (-0.332329303,-0.33939603), (3.73278689,5.4277935)], [(-0.365511209,2.3009584), (-0.927284121,2.99728084), (1.25863123,-1.16865635), (1.62553263,-4.39500284)]]> : tensor<4x4xcomplex<f32>>
    %1 = stablehlo.constant dense<[[(2.38373899,-0.784328401)], [(0.560400665,-2.78799081)], [(-5.61439896,-1.56043386)], [(4.73302555,-5.01619768)]]> : tensor<4x1xcomplex<f32>>
    return %0, %1 : tensor<4x4xcomplex<f32>>, tensor<4x1xcomplex<f32>>
  }
  func.func private @expected() -> tensor<4x1xcomplex<f32>> {
    %0 = stablehlo.constant dense<[[(-0.603416204,0.428950876)], [(-1.44485509,1.09551668)], [(24.8797417,-6.34990501)], [(-9.01238632,-0.409644127)]]> : tensor<4x1xcomplex<f32>>
    return %0 : tensor<4x1xcomplex<f32>>
  }
}
