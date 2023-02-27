module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<2x5x5xf32>
    %1 = call @expected() : () -> tensor<2x5x5xf32>
    %2 = stablehlo.constant dense<1> : tensor<i32>
    %3 = stablehlo.constant dense<2> : tensor<i32>
    %4 = stablehlo.constant dense<5> : tensor<i32>
    %5 = stablehlo.custom_call @lapack_spotrf(%2, %3, %4, %0) {api_version = 2 : i32, operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[1, 2, 0]> : tensor<3xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 3, operand_tuple_indices = []>], result_layouts = [dense<[1, 2, 0]> : tensor<3xindex>, dense<0> : tensor<1xindex>]} : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<2x5x5xf32>) -> tuple<tensor<2x5x5xf32>, tensor<2xi32>>
    %6 = stablehlo.get_tuple_element %5[0] : (tuple<tensor<2x5x5xf32>, tensor<2xi32>>) -> tensor<2x5x5xf32>
    %7 = stablehlo.get_tuple_element %5[1] : (tuple<tensor<2x5x5xf32>, tensor<2xi32>>) -> tensor<2xi32>
    %8 = stablehlo.constant dense<0> : tensor<i32>
    %9 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<i32>) -> tensor<2xi32>
    %10 = stablehlo.compare  EQ, %7, %9,  SIGNED : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0] : (tensor<2xi1>) -> tensor<2x1x1xi1>
    %12 = stablehlo.constant dense<0x7FC00000> : tensor<f32>
    %13 = stablehlo.broadcast_in_dim %12, dims = [] : (tensor<f32>) -> tensor<2x5x5xf32>
    %14 = stablehlo.broadcast_in_dim %11, dims = [0, 1, 2] : (tensor<2x1x1xi1>) -> tensor<2x5x5xi1>
    %15 = stablehlo.select %14, %6, %13 : tensor<2x5x5xi1>, tensor<2x5x5xf32>
    %16 = stablehlo.custom_call @check.eq(%15, %1) : (tensor<2x5x5xf32>, tensor<2x5x5xf32>) -> tensor<i1>
    return %16 : tensor<i1>
  }
  func.func private @inputs() -> tensor<2x5x5xf32> {
    %0 = stablehlo.constant dense<[[[4.124260e+01, 12.0512629, 8.71879863, -5.20269537, -37.1940117], [12.0512629, 16.294733, -11.1725168, -0.358131289, -0.820867777], [8.71879863, -11.1725168, 35.9998322, 3.555691, -22.6960621], [-5.20269537, -0.358131289, 3.555691, 4.51707649, 7.227400e+00], [-37.1940117, -0.820867777, -22.696064, 7.227400e+00, 51.2895775]], [[10.7677689, -0.302646518, -3.61526346, 8.7007122, -8.75106525], [-0.302646518, 10.3487902, -0.46989274, -10.2065544, -2.69445491], [-3.61526346, -0.46989274, 30.8825302, -36.8722878, 6.97905207], [8.7007122, -10.2065544, -36.8722878, 6.210220e+01, -21.8459301], [-8.75106525, -2.69445515, 6.97905254, -21.8459301, 45.1717415]]]> : tensor<2x5x5xf32>
    return %0 : tensor<2x5x5xf32>
  }
  func.func private @expected() -> tensor<2x5x5xf32> {
    %0 = stablehlo.constant dense<[[[6.42203998, 12.0512629, 8.71879863, -5.20269537, -37.1940117], [1.87654757, 3.57397556, -11.1725168, -0.358131289, -0.820867777], [1.35763693, -3.83891439, 4.40674353, 3.555691, -22.6960621], [-0.810131311, 0.325161308, 1.33972442, 1.40006161, 7.227400e+00], [-5.791620e+00, 2.81126213, -0.916992962, 2.03550434, 2.20439816]], [[3.28142786, -0.302646518, -3.61526346, 8.7007122, -8.75106525], [-0.0922301263, 3.21563101, -0.46989274, -10.2065544, -2.69445491], [-1.10173488, -0.177727446, 5.44399881, -36.8722878, 6.97905207], [2.65150189, -3.09799385, -6.3375535, 2.30425382, -21.8459301], [-2.66684675, -0.914414167, 0.71241343, -5.68195677, 2.10507727]]]> : tensor<2x5x5xf32>
    return %0 : tensor<2x5x5xf32>
  }
}
