// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<8x9xf32>
    %1 = call @expected() : () -> tensor<8x9xf32>
    %2 = call @cumsum(%0) : (tensor<8x9xf32>) -> tensor<8x9xf32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<8x9xf32>, tensor<8x9xf32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<8x9xf32> {
    %0 = stablehlo.constant dense<[[1.57026935, 0.498674721, 3.12457395, 1.03478563, 0.910505414, 4.98582602, -0.0783900544, -4.256090e+00, -1.71861959], [-0.663646519, -3.02339053, -1.61709666, -3.96415782, -4.40147686, 1.00734496, -2.55806088, -0.729298055, -5.26293373], [1.27824736, -1.76403606, 2.63086438, -1.74868166, -0.230708018, -0.919561922, -0.0587415695, -0.0772727131, 0.657602965], [-2.90678334, 0.460775256, 3.68848658, 1.96552026, 1.51636684, 1.42161012, -2.39654875, 7.63749313, -0.0278683081], [-2.96210408, 0.559519768, -3.41699457, 2.57372475, 3.13115287, -0.884510099, 4.10184097, 1.25748599, -2.73227549], [1.42393124, 4.89939117, -0.436749697, 1.4321301, -1.29280937, -1.19573164, -0.966311812, 5.1226573, -0.64571023], [-0.555515289, -0.350871354, -1.32086277, -1.2682389, -3.08176184, 2.73264217, -0.315091193, -2.28265262, -4.34558678], [-2.46857905, 1.0896368, -1.3703711, -0.440513849, -1.5498302, 2.04482746, -2.35210133, 0.448786944, -0.865290343]]> : tensor<8x9xf32>
    return %0 : tensor<8x9xf32>
  }
  func.func private @expected() -> tensor<8x9xf32> {
    %0 = stablehlo.constant dense<[[-5.28418064, 2.369700e+00, 1.28185034, -0.415431738, -4.99856091, 9.1924467, -4.62340498, 7.12110949, -14.9406815], [-6.854450e+00, 1.87102532, -1.84272361, -1.45021725, -5.9090662, 4.20662117, -4.54501438, 1.137720e+01, -13.2220612], [-6.19080352, 4.89441586, -0.225626945, 2.51394057, -1.50758934, 3.19927621, -1.98695374, 12.1064978, -7.9591279], [-7.4690504, 6.65845203, -2.85649157, 4.26262236, -1.27688134, 4.11883831, -1.92821217, 12.1837711, -8.61673069], [-4.5622673, 6.19767665, -6.54497814, 2.29710197, -2.79324818, 2.69722795, 0.468336582, 4.54627752, -8.58886241], [-1.6001631, 5.63815689, -3.12798357, -0.276622653, -5.92440128, 3.581738, -3.63350439, 3.28879166, -5.85658741], [-3.02409434, 0.738765478, -2.69123387, -1.70875275, -4.6315918, 4.77746964, -2.66719246, -1.83386564, -5.21087694], [-2.46857905, 1.0896368, -1.3703711, -0.440513849, -1.5498302, 2.04482746, -2.35210133, 0.448786944, -0.865290343]]> : tensor<8x9xf32>
    return %0 : tensor<8x9xf32>
  }
  func.func private @cumsum(%arg0: tensor<8x9xf32>) -> tensor<8x9xf32> {
    %0 = stablehlo.reverse %arg0, dims = [0] : tensor<8x9xf32>
    %1 = "stablehlo.slice"(%0) {limit_indices = dense<[7, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<8x9xf32>) -> tensor<4x9xf32>
    %2 = "stablehlo.slice"(%0) {limit_indices = dense<[8, 9]> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<8x9xf32>) -> tensor<4x9xf32>
    %3 = stablehlo.add %1, %2 : tensor<4x9xf32>
    %4 = "stablehlo.slice"(%3) {limit_indices = dense<[3, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<4x9xf32>) -> tensor<2x9xf32>
    %5 = "stablehlo.slice"(%3) {limit_indices = dense<[4, 9]> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<4x9xf32>) -> tensor<2x9xf32>
    %6 = stablehlo.add %4, %5 : tensor<2x9xf32>
    %7 = "stablehlo.slice"(%6) {limit_indices = dense<[1, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<2x9xf32>) -> tensor<1x9xf32>
    %8 = "stablehlo.slice"(%6) {limit_indices = dense<[2, 9]> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<2x9xf32>) -> tensor<1x9xf32>
    %9 = stablehlo.add %7, %8 : tensor<1x9xf32>
    %10 = "stablehlo.slice"(%9) {limit_indices = dense<[0, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<1x9xf32>) -> tensor<0x9xf32>
    %11 = "stablehlo.slice"(%6) {limit_indices = dense<[2, 9]> : tensor<2xi64>, start_indices = dense<[2, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<2x9xf32>) -> tensor<0x9xf32>
    %12 = stablehlo.add %10, %11 : tensor<0x9xf32>
    %13 = "stablehlo.slice"(%6) {limit_indices = dense<[1, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<2x9xf32>) -> tensor<1x9xf32>
    %14 = stablehlo.concatenate %13, %12, dim = 0 : (tensor<1x9xf32>, tensor<0x9xf32>) -> tensor<1x9xf32>
    %15 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %16 = stablehlo.pad %14, %15, low = [0, 0], high = [1, 0], interior = [1, 0] : (tensor<1x9xf32>, tensor<f32>) -> tensor<2x9xf32>
    %17 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %18 = stablehlo.pad %9, %17, low = [1, 0], high = [0, 0], interior = [1, 0] : (tensor<1x9xf32>, tensor<f32>) -> tensor<2x9xf32>
    %19 = stablehlo.add %16, %18 : tensor<2x9xf32>
    %20 = "stablehlo.slice"(%19) {limit_indices = dense<[1, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<2x9xf32>) -> tensor<1x9xf32>
    %21 = "stablehlo.slice"(%3) {limit_indices = dense<[4, 9]> : tensor<2xi64>, start_indices = dense<[2, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<4x9xf32>) -> tensor<1x9xf32>
    %22 = stablehlo.add %20, %21 : tensor<1x9xf32>
    %23 = "stablehlo.slice"(%3) {limit_indices = dense<[1, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<4x9xf32>) -> tensor<1x9xf32>
    %24 = stablehlo.concatenate %23, %22, dim = 0 : (tensor<1x9xf32>, tensor<1x9xf32>) -> tensor<2x9xf32>
    %25 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %26 = stablehlo.pad %24, %25, low = [0, 0], high = [1, 0], interior = [1, 0] : (tensor<2x9xf32>, tensor<f32>) -> tensor<4x9xf32>
    %27 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %28 = stablehlo.pad %19, %27, low = [1, 0], high = [0, 0], interior = [1, 0] : (tensor<2x9xf32>, tensor<f32>) -> tensor<4x9xf32>
    %29 = stablehlo.add %26, %28 : tensor<4x9xf32>
    %30 = "stablehlo.slice"(%29) {limit_indices = dense<[3, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<4x9xf32>) -> tensor<3x9xf32>
    %31 = "stablehlo.slice"(%0) {limit_indices = dense<[8, 9]> : tensor<2xi64>, start_indices = dense<[2, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<8x9xf32>) -> tensor<3x9xf32>
    %32 = stablehlo.add %30, %31 : tensor<3x9xf32>
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
