module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0:2 = call @inputs() : () -> (tensor<4x4xcomplex<f32>>, tensor<4x1xcomplex<f32>>)
    %1 = call @expected() : () -> tensor<4x1xcomplex<f32>>
    %2 = chlo.conj %0#0 : tensor<4x4xcomplex<f32>> -> tensor<4x4xcomplex<f32>>
    %3 = stablehlo.constant dense<(1.000000e+00,0.000000e+00)> : tensor<complex<f32>>
    %4 = stablehlo.constant dense<1> : tensor<i32>
    %5 = stablehlo.constant dense<0> : tensor<i32>
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
    %0 = stablehlo.constant dense<[[(-1.59752703,5.56757307), (2.31823397,0.100249283), (4.97805834,0.639441133), (0.224809214,0.332314551)], [(-0.254555106,1.17679703), (5.34579754,-2.89770103), (-0.346758276,3.79027414), (-2.88891459,2.52692389)], [(-1.94236255,0.280263811), (4.76986599,0.994465529), (-3.77877402,0.589053392), (0.233623058,1.79179549)], [(2.19535089,-1.60195363), (2.65920663,-0.89382857), (0.423473388,0.189313531), (1.02549827,0.231334671)]]> : tensor<4x4xcomplex<f32>>
    %1 = stablehlo.constant dense<[[(-8.62764359,-0.74485892)], [(0.991535782,-4.05390692)], [(0.637544453,0.751244962)], [(-0.874452531,-1.70285988)]]> : tensor<4x1xcomplex<f32>>
    return %0, %1 : tensor<4x4xcomplex<f32>>, tensor<4x1xcomplex<f32>>
  }
  func.func private @expected() -> tensor<4x1xcomplex<f32>> {
    %0 = stablehlo.constant dense<[[(-0.575737894,-0.393365681)], [(-0.599522054,-2.32178259)], [(-1.02239633,6.729860e-02)], [(-0.454972923,-1.76315355)]]> : tensor<4x1xcomplex<f32>>
    return %0 : tensor<4x1xcomplex<f32>>
  }
}
