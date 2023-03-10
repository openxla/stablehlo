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
    %0 = stablehlo.constant dense<[[-0.516386032, -4.69376278, 1.52112198, 4.05397034, -3.29195356, 4.86984587, 1.82326937, -1.60891724, -0.577785432], [-0.758550107, 1.69321144, -0.00915546901, 3.01130462, -3.44816279, 2.79591393, -1.48233068, 7.381420e-01, -0.955493092], [-1.6551671, -1.77591515, -8.39676761, 2.89952898, 4.06108856, -6.62465429, -5.36294842, -2.01264596, -2.50611639], [0.0494485199, 3.9402771, -1.35655165, 5.62359285, 0.587394118, -2.62389517, -3.80574322, -2.47952676, 2.5006156], [-4.55041647, -2.2147491, 1.10280836, 3.60253811, -4.21449137, 2.15035033, -1.17413747, 1.01869929, 1.37833536], [1.70794797, -4.141660e+00, -3.98779249, 1.86026859, -3.254330e+00, -1.26010954, -0.0639877617, 4.07550049, 4.7545476], [-0.171428666, 0.958532869, 1.0267899, -6.48032283, 1.58425522, 3.64146233, 5.61766815, 1.59009778, -2.01838231], [-9.273170e-01, 7.40547466, -1.34822798, 2.23499012, 2.79915023, -3.21878052, 1.70609844, 3.8728838, 3.81931067]]> : tensor<8x9xf32>
    return %0 : tensor<8x9xf32>
  }
  func.func private @expected() -> tensor<8x9xf32> {
    %0 = stablehlo.constant dense<[[-0.516386032, -4.69376278, 1.52112198, 4.05397034, -3.29195356, 4.86984587, 1.82326937, -1.60891724, -0.577785432], [0.391704679, -7.94753265, -0.0139265852, 12.2077398, 11.3511915, 13.6156702, -2.70268822, -1.18760943, 0.552069962], [-0.648336708, 14.1141434, 0.1169383, 35.3966942, 46.0981941, -90.1991043, 14.4943771, 2.39023733, -1.3835516], [-0.0320592895, 55.613636, -0.158632845, 199.05661, 27.0778065, 2.366730e+02, -55.1618767, -5.9266572, -3.45973063], [0.145883113, -123.17025, -0.174941629, 7.171090e+02, -114.119179, 508.929871, 64.7676239, -6.03748131, -4.76866913], [0.249160767, 510.129303, 0.697630882, 1334.01538, 371.38147, -641.307373, -4.14433527, -24.6057587, -22.672863], [-4.271330e-02, 488.975708, 0.716320335, -8644.85058, 588.363037, -2335.29663, -2.328150e+01, -39.1255608, 45.7625046], [0.0396087691, 3621.09717, -0.965763151, -19321.1563, 1646.9165, 7516.80761, -39.7205353, -151.528763, 174.781219]]> : tensor<8x9xf32>
    return %0 : tensor<8x9xf32>
  }
  func.func private @cumprod(%arg0: tensor<8x9xf32>) -> tensor<8x9xf32> {
    %0 = "stablehlo.slice"(%arg0) {limit_indices = dense<[7, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<8x9xf32>) -> tensor<4x9xf32>
    %1 = "stablehlo.slice"(%arg0) {limit_indices = dense<[8, 9]> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<8x9xf32>) -> tensor<4x9xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<4x9xf32>
    %3 = "stablehlo.slice"(%2) {limit_indices = dense<[3, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<4x9xf32>) -> tensor<2x9xf32>
    %4 = "stablehlo.slice"(%2) {limit_indices = dense<[4, 9]> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<4x9xf32>) -> tensor<2x9xf32>
    %5 = stablehlo.multiply %3, %4 : tensor<2x9xf32>
    %6 = "stablehlo.slice"(%5) {limit_indices = dense<[1, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<2x9xf32>) -> tensor<1x9xf32>
    %7 = "stablehlo.slice"(%5) {limit_indices = dense<[2, 9]> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<2x9xf32>) -> tensor<1x9xf32>
    %8 = stablehlo.multiply %6, %7 : tensor<1x9xf32>
    %9 = "stablehlo.slice"(%8) {limit_indices = dense<[0, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<1x9xf32>) -> tensor<0x9xf32>
    %10 = "stablehlo.slice"(%5) {limit_indices = dense<[2, 9]> : tensor<2xi64>, start_indices = dense<[2, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<2x9xf32>) -> tensor<0x9xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<0x9xf32>
    %12 = "stablehlo.slice"(%5) {limit_indices = dense<[1, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<2x9xf32>) -> tensor<1x9xf32>
    %13 = stablehlo.concatenate %12, %11, dim = 0 : (tensor<1x9xf32>, tensor<0x9xf32>) -> tensor<1x9xf32>
    %14 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %15 = stablehlo.pad %13, %14, low = [0, 0], high = [1, 0], interior = [1, 0] : (tensor<1x9xf32>, tensor<f32>) -> tensor<2x9xf32>
    %16 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %17 = stablehlo.pad %8, %16, low = [1, 0], high = [0, 0], interior = [1, 0] : (tensor<1x9xf32>, tensor<f32>) -> tensor<2x9xf32>
    %18 = stablehlo.add %15, %17 : tensor<2x9xf32>
    %19 = "stablehlo.slice"(%18) {limit_indices = dense<[1, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<2x9xf32>) -> tensor<1x9xf32>
    %20 = "stablehlo.slice"(%2) {limit_indices = dense<[4, 9]> : tensor<2xi64>, start_indices = dense<[2, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<4x9xf32>) -> tensor<1x9xf32>
    %21 = stablehlo.multiply %19, %20 : tensor<1x9xf32>
    %22 = "stablehlo.slice"(%2) {limit_indices = dense<[1, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<4x9xf32>) -> tensor<1x9xf32>
    %23 = stablehlo.concatenate %22, %21, dim = 0 : (tensor<1x9xf32>, tensor<1x9xf32>) -> tensor<2x9xf32>
    %24 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %25 = stablehlo.pad %23, %24, low = [0, 0], high = [1, 0], interior = [1, 0] : (tensor<2x9xf32>, tensor<f32>) -> tensor<4x9xf32>
    %26 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %27 = stablehlo.pad %18, %26, low = [1, 0], high = [0, 0], interior = [1, 0] : (tensor<2x9xf32>, tensor<f32>) -> tensor<4x9xf32>
    %28 = stablehlo.add %25, %27 : tensor<4x9xf32>
    %29 = "stablehlo.slice"(%28) {limit_indices = dense<[3, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<4x9xf32>) -> tensor<3x9xf32>
    %30 = "stablehlo.slice"(%arg0) {limit_indices = dense<[8, 9]> : tensor<2xi64>, start_indices = dense<[2, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<8x9xf32>) -> tensor<3x9xf32>
    %31 = stablehlo.multiply %29, %30 : tensor<3x9xf32>
    %32 = "stablehlo.slice"(%arg0) {limit_indices = dense<[1, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<8x9xf32>) -> tensor<1x9xf32>
    %33 = stablehlo.concatenate %32, %31, dim = 0 : (tensor<1x9xf32>, tensor<3x9xf32>) -> tensor<4x9xf32>
    %34 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %35 = stablehlo.pad %33, %34, low = [0, 0], high = [1, 0], interior = [1, 0] : (tensor<4x9xf32>, tensor<f32>) -> tensor<8x9xf32>
    %36 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %37 = stablehlo.pad %28, %36, low = [1, 0], high = [0, 0], interior = [1, 0] : (tensor<4x9xf32>, tensor<f32>) -> tensor<8x9xf32>
    %38 = stablehlo.add %35, %37 : tensor<8x9xf32>
    return %38 : tensor<8x9xf32>
  }
}
