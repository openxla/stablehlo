// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<8x9xf32>
    %1 = call @expected() : () -> tensor<8x9xf32>
    %2 = call @cummin(%0) : (tensor<8x9xf32>) -> tensor<8x9xf32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<8x9xf32>, tensor<8x9xf32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<8x9xf32> {
    %0 = stablehlo.constant dense<[[6.10286379, -7.55283641, 4.81546545, -0.275943786, 4.2643795, 2.4054687, 0.920640468, -4.68816042, 4.16136789], [1.96532786, 1.07312596, 0.959175586, 1.4127574, 3.77626681, -2.16750836, -2.66594076, 9.87853622, -1.41031206], [-2.90562558, -6.92489433, -3.78238487, -0.0983529165, -3.18020034, 0.74258548, -3.31251335, -1.50061619, -0.690689206], [0.318664074, -2.01927662, 1.56834769, 0.444916934, 0.848639488, -1.22545588, 1.68221557, -0.639991164, 3.40163589], [0.1287487, -3.50567985, -2.48051667, 7.36464643, -0.553622246, -2.13473606, 0.538260937, 0.382442355, -1.04872513], [2.1969552, 2.39613724, 2.75187087, 3.80963373, 0.724594474, 0.484853566, -6.78865576, 5.30916548, 3.67710304], [-1.290295, 0.534302592, 5.54923296, 1.22981727, -4.33742762, 3.99035096, 5.48597717, -2.89982438, -6.36005831], [3.74490714, 6.40041304, 4.82009792, 2.20148969, -6.25037288, -2.49714422, 1.17261744, 0.309043527, -1.35576725]]> : tensor<8x9xf32>
    return %0 : tensor<8x9xf32>
  }
  func.func private @expected() -> tensor<8x9xf32> {
    %0 = stablehlo.constant dense<[[6.10286379, -7.55283641, -7.55283641, -7.55283641, -7.55283641, -7.55283641, -7.55283641, -7.55283641, -7.55283641], [1.96532786, 1.07312596, 0.959175586, 0.959175586, 0.959175586, -2.16750836, -2.66594076, -2.66594076, -2.66594076], [-2.90562558, -6.92489433, -6.92489433, -6.92489433, -6.92489433, -6.92489433, -6.92489433, -6.92489433, -6.92489433], [0.318664074, -2.01927662, -2.01927662, -2.01927662, -2.01927662, -2.01927662, -2.01927662, -2.01927662, -2.01927662], [0.1287487, -3.50567985, -3.50567985, -3.50567985, -3.50567985, -3.50567985, -3.50567985, -3.50567985, -3.50567985], [2.1969552, 2.1969552, 2.1969552, 2.1969552, 0.724594474, 0.484853566, -6.78865576, -6.78865576, -6.78865576], [-1.290295, -1.290295, -1.290295, -1.290295, -4.33742762, -4.33742762, -4.33742762, -4.33742762, -6.36005831], [3.74490714, 3.74490714, 3.74490714, 2.20148969, -6.25037288, -6.25037288, -6.25037288, -6.25037288, -6.25037288]]> : tensor<8x9xf32>
    return %0 : tensor<8x9xf32>
  }
  func.func private @cummin(%arg0: tensor<8x9xf32>) -> tensor<8x9xf32> {
    %0 = "stablehlo.slice"(%arg0) {limit_indices = dense<8> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<[1, 2]> : tensor<2xi64>} : (tensor<8x9xf32>) -> tensor<8x4xf32>
    %1 = "stablehlo.slice"(%arg0) {limit_indices = dense<[8, 9]> : tensor<2xi64>, start_indices = dense<[0, 1]> : tensor<2xi64>, strides = dense<[1, 2]> : tensor<2xi64>} : (tensor<8x9xf32>) -> tensor<8x4xf32>
    %2 = stablehlo.minimum %0, %1 : tensor<8x4xf32>
    %3 = "stablehlo.slice"(%2) {limit_indices = dense<[8, 3]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<[1, 2]> : tensor<2xi64>} : (tensor<8x4xf32>) -> tensor<8x2xf32>
    %4 = "stablehlo.slice"(%2) {limit_indices = dense<[8, 4]> : tensor<2xi64>, start_indices = dense<[0, 1]> : tensor<2xi64>, strides = dense<[1, 2]> : tensor<2xi64>} : (tensor<8x4xf32>) -> tensor<8x2xf32>
    %5 = stablehlo.minimum %3, %4 : tensor<8x2xf32>
    %6 = "stablehlo.slice"(%5) {limit_indices = dense<[8, 1]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<[1, 2]> : tensor<2xi64>} : (tensor<8x2xf32>) -> tensor<8x1xf32>
    %7 = "stablehlo.slice"(%5) {limit_indices = dense<[8, 2]> : tensor<2xi64>, start_indices = dense<[0, 1]> : tensor<2xi64>, strides = dense<[1, 2]> : tensor<2xi64>} : (tensor<8x2xf32>) -> tensor<8x1xf32>
    %8 = stablehlo.minimum %6, %7 : tensor<8x1xf32>
    %9 = "stablehlo.slice"(%8) {limit_indices = dense<[8, 0]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<8x1xf32>) -> tensor<8x0xf32>
    %10 = "stablehlo.slice"(%5) {limit_indices = dense<[8, 2]> : tensor<2xi64>, start_indices = dense<[0, 2]> : tensor<2xi64>, strides = dense<[1, 2]> : tensor<2xi64>} : (tensor<8x2xf32>) -> tensor<8x0xf32>
    %11 = stablehlo.minimum %9, %10 : tensor<8x0xf32>
    %12 = "stablehlo.slice"(%5) {limit_indices = dense<[8, 1]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<8x2xf32>) -> tensor<8x1xf32>
    %13 = stablehlo.concatenate %12, %11, dim = 1 : (tensor<8x1xf32>, tensor<8x0xf32>) -> tensor<8x1xf32>
    %14 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %15 = stablehlo.pad %13, %14, low = [0, 0], high = [0, 1], interior = [0, 1] : (tensor<8x1xf32>, tensor<f32>) -> tensor<8x2xf32>
    %16 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %17 = stablehlo.pad %8, %16, low = [0, 1], high = [0, 0], interior = [0, 1] : (tensor<8x1xf32>, tensor<f32>) -> tensor<8x2xf32>
    %18 = stablehlo.add %15, %17 : tensor<8x2xf32>
    %19 = "stablehlo.slice"(%18) {limit_indices = dense<[8, 1]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<8x2xf32>) -> tensor<8x1xf32>
    %20 = "stablehlo.slice"(%2) {limit_indices = dense<[8, 4]> : tensor<2xi64>, start_indices = dense<[0, 2]> : tensor<2xi64>, strides = dense<[1, 2]> : tensor<2xi64>} : (tensor<8x4xf32>) -> tensor<8x1xf32>
    %21 = stablehlo.minimum %19, %20 : tensor<8x1xf32>
    %22 = "stablehlo.slice"(%2) {limit_indices = dense<[8, 1]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<8x4xf32>) -> tensor<8x1xf32>
    %23 = stablehlo.concatenate %22, %21, dim = 1 : (tensor<8x1xf32>, tensor<8x1xf32>) -> tensor<8x2xf32>
    %24 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %25 = stablehlo.pad %23, %24, low = [0, 0], high = [0, 1], interior = [0, 1] : (tensor<8x2xf32>, tensor<f32>) -> tensor<8x4xf32>
    %26 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %27 = stablehlo.pad %18, %26, low = [0, 1], high = [0, 0], interior = [0, 1] : (tensor<8x2xf32>, tensor<f32>) -> tensor<8x4xf32>
    %28 = stablehlo.add %25, %27 : tensor<8x4xf32>
    %29 = "stablehlo.slice"(%arg0) {limit_indices = dense<[8, 9]> : tensor<2xi64>, start_indices = dense<[0, 2]> : tensor<2xi64>, strides = dense<[1, 2]> : tensor<2xi64>} : (tensor<8x9xf32>) -> tensor<8x4xf32>
    %30 = stablehlo.minimum %28, %29 : tensor<8x4xf32>
    %31 = "stablehlo.slice"(%arg0) {limit_indices = dense<[8, 1]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<8x9xf32>) -> tensor<8x1xf32>
    %32 = stablehlo.concatenate %31, %30, dim = 1 : (tensor<8x1xf32>, tensor<8x4xf32>) -> tensor<8x5xf32>
    %33 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.pad %32, %33, low = [0, 0], high = [0, 0], interior = [0, 1] : (tensor<8x5xf32>, tensor<f32>) -> tensor<8x9xf32>
    %35 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %36 = stablehlo.pad %28, %35, low = [0, 1], high = [0, 1], interior = [0, 1] : (tensor<8x4xf32>, tensor<f32>) -> tensor<8x9xf32>
    %37 = stablehlo.add %34, %36 : tensor<8x9xf32>
    return %37 : tensor<8x9xf32>
  }
}
