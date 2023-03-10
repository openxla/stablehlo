// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<8x9xf32>
    %1 = call @expected() : () -> tensor<8x9xf32>
    %2 = call @cumprod(%0) : (tensor<8x9xf32>) -> tensor<8x9xf32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<8x9xf32>, tensor<8x9xf32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<8x9xf32> {
    %0 = stablehlo.constant dense<[[-2.09422636, 0.380081028, -0.781425654, 5.48995972, -2.64779496, -4.87315941, -0.462664813, -3.04496455, 4.05275583], [-1.73371768, 8.208870e-01, 2.75937986, 0.224293321, -3.71326518, 2.58386803, 1.45708978, -1.664990e+00, 1.60126698], [1.79414046, 2.55974531, 1.29430878, 1.88928807, -7.93745947, 0.219101891, 1.10423291, -2.50036979, -4.55070734], [2.08415461, 2.72951078, -0.692231536, 1.52651727, 2.83530426, -4.5625124, -0.535679698, 1.77830541, -3.96226072], [-3.31649065, 0.422209948, 0.0209581442, 4.8679738, -0.24322173, -2.75813556, -0.725915849, 1.34848034, 0.648082315], [-3.87982202, -6.49089241, 1.29380846, 5.22685957, 2.43060613, -5.9127779, -4.68131638, -3.10234046, -0.148022592], [1.74197829, 6.33971739, -1.39046419, -3.86233926, -6.03123426, -0.155255735, 1.05866158, 2.14641809, -1.25763392], [-3.09568191, -1.06392372, -2.389480e+00, -0.27208522, -1.20874333, 0.998996555, 1.9902693, 5.5722785, 1.70640945]]> : tensor<8x9xf32>
    return %0 : tensor<8x9xf32>
  }
  func.func private @expected() -> tensor<8x9xf32> {
    %0 = stablehlo.constant dense<[[-942.058776, 4.029530e+01, 0.174050167, 94.9576263, 953.627319, -31.8383026, 2.8552351, 1127.93909, 24.0897236], [449.836182, 106.017654, -0.222734138, 17.2965984, -360.159058, 6.53340101, -6.17128276, -370.427673, 5.94403553], [-259.463348, 129.150116, -0.0807189121, 77.1159744, 96.9925537, 2.52853513, -4.23534822, 222.480423, 3.71208286], [-144.617065, 50.4542847, -0.0623644963, 40.8174782, -12.2195978, 11.540453, -3.83555698, -88.9790115, -0.815715551], [-69.3888397, 18.4847355, 0.0900919586, 26.7389565, -4.3098011, -2.5294075, 7.16016864, -50.0358429, 0.205871239], [20.922369, 43.7809105, 4.29866123, 5.49283075, 17.7196388, 0.917071521, -9.8636341, -37.1053543, 0.317662179], [-5.39261055, -6.74497557, 3.32248664, 1.05088544, 7.29021406, -0.155099943, 2.10702157, 11.9604397, -2.14603853], [-3.09568191, -1.06392372, -2.389480e+00, -0.27208522, -1.20874333, 0.998996555, 1.9902693, 5.5722785, 1.70640945]]> : tensor<8x9xf32>
    return %0 : tensor<8x9xf32>
  }
  func.func private @cumprod(%arg0: tensor<8x9xf32>) -> tensor<8x9xf32> {
    %0 = stablehlo.reverse %arg0, dims = [0] : tensor<8x9xf32>
    %1 = "stablehlo.slice"(%0) {limit_indices = dense<[7, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<8x9xf32>) -> tensor<4x9xf32>
    %2 = "stablehlo.slice"(%0) {limit_indices = dense<[8, 9]> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<8x9xf32>) -> tensor<4x9xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<4x9xf32>
    %4 = "stablehlo.slice"(%3) {limit_indices = dense<[3, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<4x9xf32>) -> tensor<2x9xf32>
    %5 = "stablehlo.slice"(%3) {limit_indices = dense<[4, 9]> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<4x9xf32>) -> tensor<2x9xf32>
    %6 = stablehlo.multiply %4, %5 : tensor<2x9xf32>
    %7 = "stablehlo.slice"(%6) {limit_indices = dense<[1, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<2x9xf32>) -> tensor<1x9xf32>
    %8 = "stablehlo.slice"(%6) {limit_indices = dense<[2, 9]> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<2x9xf32>) -> tensor<1x9xf32>
    %9 = stablehlo.multiply %7, %8 : tensor<1x9xf32>
    %10 = "stablehlo.slice"(%9) {limit_indices = dense<[0, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<1x9xf32>) -> tensor<0x9xf32>
    %11 = "stablehlo.slice"(%6) {limit_indices = dense<[2, 9]> : tensor<2xi64>, start_indices = dense<[2, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<2x9xf32>) -> tensor<0x9xf32>
    %12 = stablehlo.multiply %10, %11 : tensor<0x9xf32>
    %13 = "stablehlo.slice"(%6) {limit_indices = dense<[1, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<2x9xf32>) -> tensor<1x9xf32>
    %14 = stablehlo.concatenate %13, %12, dim = 0 : (tensor<1x9xf32>, tensor<0x9xf32>) -> tensor<1x9xf32>
    %15 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %16 = stablehlo.pad %14, %15, low = [0, 0], high = [1, 0], interior = [1, 0] : (tensor<1x9xf32>, tensor<f32>) -> tensor<2x9xf32>
    %17 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %18 = stablehlo.pad %9, %17, low = [1, 0], high = [0, 0], interior = [1, 0] : (tensor<1x9xf32>, tensor<f32>) -> tensor<2x9xf32>
    %19 = stablehlo.add %16, %18 : tensor<2x9xf32>
    %20 = "stablehlo.slice"(%19) {limit_indices = dense<[1, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<2x9xf32>) -> tensor<1x9xf32>
    %21 = "stablehlo.slice"(%3) {limit_indices = dense<[4, 9]> : tensor<2xi64>, start_indices = dense<[2, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<4x9xf32>) -> tensor<1x9xf32>
    %22 = stablehlo.multiply %20, %21 : tensor<1x9xf32>
    %23 = "stablehlo.slice"(%3) {limit_indices = dense<[1, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<4x9xf32>) -> tensor<1x9xf32>
    %24 = stablehlo.concatenate %23, %22, dim = 0 : (tensor<1x9xf32>, tensor<1x9xf32>) -> tensor<2x9xf32>
    %25 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %26 = stablehlo.pad %24, %25, low = [0, 0], high = [1, 0], interior = [1, 0] : (tensor<2x9xf32>, tensor<f32>) -> tensor<4x9xf32>
    %27 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %28 = stablehlo.pad %19, %27, low = [1, 0], high = [0, 0], interior = [1, 0] : (tensor<2x9xf32>, tensor<f32>) -> tensor<4x9xf32>
    %29 = stablehlo.add %26, %28 : tensor<4x9xf32>
    %30 = "stablehlo.slice"(%29) {limit_indices = dense<[3, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<4x9xf32>) -> tensor<3x9xf32>
    %31 = "stablehlo.slice"(%0) {limit_indices = dense<[8, 9]> : tensor<2xi64>, start_indices = dense<[2, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<8x9xf32>) -> tensor<3x9xf32>
    %32 = stablehlo.multiply %30, %31 : tensor<3x9xf32>
    %33 = "stablehlo.slice"(%0) {limit_indices = dense<[1, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<8x9xf32>) -> tensor<1x9xf32>
    %34 = stablehlo.concatenate %33, %32, dim = 0 : (tensor<1x9xf32>, tensor<3x9xf32>) -> tensor<4x9xf32>
    %35 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %36 = stablehlo.pad %34, %35, low = [0, 0], high = [1, 0], interior = [1, 0] : (tensor<4x9xf32>, tensor<f32>) -> tensor<8x9xf32>
    %37 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %38 = stablehlo.pad %29, %37, low = [1, 0], high = [0, 0], interior = [1, 0] : (tensor<4x9xf32>, tensor<f32>) -> tensor<8x9xf32>
    %39 = stablehlo.add %36, %38 : tensor<8x9xf32>
    %40 = stablehlo.reverse %39, dims = [0] : tensor<8x9xf32>
    return %40 : tensor<8x9xf32>
  }
}
