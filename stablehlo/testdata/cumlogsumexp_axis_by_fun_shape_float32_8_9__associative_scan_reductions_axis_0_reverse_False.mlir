// RUN-DISABLED: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<8x9xf32>
    %1 = call @expected() : () -> tensor<8x9xf32>
    %2 = call @cumlogsumexp(%0) : (tensor<8x9xf32>) -> tensor<8x9xf32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<8x9xf32>, tensor<8x9xf32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<8x9xf32> {
    %0 = stablehlo.constant dense<[[-7.1557703, 0.788234472, 1.2348038, -1.98865879, 1.48776627, 1.94569707, 2.37920785, 0.497390181, 1.58107507], [3.23350048, -1.44753039, -0.547469378, 0.593297124, 0.230663791, -0.767583966, -2.03303456, -2.94393921, -1.6824317], [-0.395836532, 1.97770965, 1.17376947, 4.34619713, -5.720222, -1.16927159, 7.94830608, -1.90525365, -2.6565938], [-3.80136395, 3.44854879, 1.35449398, -4.20873594, 2.84162831, 4.24899578, 4.49400902, 0.811294972, 5.0623312], [0.602592766, 3.36404777, -3.77222037, 0.308629572, -0.426342517, -5.45122194, -1.94540536, 2.85654593, -3.02575016], [-0.0889500901, 0.689754248, 2.92732906, 1.5455339, 5.03851128, -9.63094139, 4.86661863, 1.21049035, -1.12018597], [-1.44973421, 0.991479694, 4.02561855, 1.99775958, 1.65332413, -1.20078361, 0.491165251, 0.587167263, 4.20799303], [-3.99268913, -4.99328613, 0.721381247, 4.21440935, -0.88381958, -1.43108499, 5.595500e+00, 1.02258921, -2.69825149]]> : tensor<8x9xf32>
    return %0 : tensor<8x9xf32>
  }
  func.func private @expected() -> tensor<8x9xf32> {
    %0 = stablehlo.constant dense<[[-7.1557703, 0.788234472, 1.2348038, -1.98865879, 1.48776627, 1.94569707, 2.37920785, 0.497390181, 1.58107507], [3.23353124, 0.889807105, 1.39031518, 0.666199862, 1.73811793, 2.00990939, 2.39126277, 0.528910279, 1.61861551], [3.25971818, 2.268080e+00, 1.98103952, 4.3711071, 1.73869443, 2.05068636, 7.95215892, 6.129490e-01, 1.63242877], [3.26057577, 3.71638632, 2.40920162, 4.37129545, 3.12823176, 4.35424805, 7.98316145, 1.41017878, 5.09420776], [3.32831812, 4.24880219, 2.41126704, 4.38835239, 3.15642405, 4.35430336, 7.983210e+00, 3.06796026, 5.09450531], [3.36059332, 4.27687073, 3.39537215, 4.44497967, 5.18024683, 4.35430384, 8.02656459, 3.21298385, 5.09650326], [3.36870551, 4.31361341, 4.45249319, 4.52797222, 5.2092185, 4.35816431, 8.02709865, 3.28286481, 5.44099188], [3.36934066, 4.31370401, 4.47617674, 5.07657814, 5.21147394, 4.361220e+00, 8.11134433, 3.38209629, 5.4412837]]> : tensor<8x9xf32>
    return %0 : tensor<8x9xf32>
  }
  func.func private @cumlogsumexp(%arg0: tensor<8x9xf32>) -> tensor<8x9xf32> {
    %0 = "stablehlo.slice"(%arg0) {limit_indices = dense<[7, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<8x9xf32>) -> tensor<4x9xf32>
    %1 = "stablehlo.slice"(%arg0) {limit_indices = dense<[8, 9]> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<8x9xf32>) -> tensor<4x9xf32>
    %2 = call @logaddexp(%0, %1) : (tensor<4x9xf32>, tensor<4x9xf32>) -> tensor<4x9xf32>
    %3 = "stablehlo.slice"(%2) {limit_indices = dense<[3, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<4x9xf32>) -> tensor<2x9xf32>
    %4 = "stablehlo.slice"(%2) {limit_indices = dense<[4, 9]> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<4x9xf32>) -> tensor<2x9xf32>
    %5 = call @logaddexp_0(%3, %4) : (tensor<2x9xf32>, tensor<2x9xf32>) -> tensor<2x9xf32>
    %6 = "stablehlo.slice"(%5) {limit_indices = dense<[1, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<2x9xf32>) -> tensor<1x9xf32>
    %7 = "stablehlo.slice"(%5) {limit_indices = dense<[2, 9]> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<2x9xf32>) -> tensor<1x9xf32>
    %8 = call @logaddexp_1(%6, %7) : (tensor<1x9xf32>, tensor<1x9xf32>) -> tensor<1x9xf32>
    %9 = "stablehlo.slice"(%8) {limit_indices = dense<[0, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<1x9xf32>) -> tensor<0x9xf32>
    %10 = "stablehlo.slice"(%5) {limit_indices = dense<[2, 9]> : tensor<2xi64>, start_indices = dense<[2, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<2x9xf32>) -> tensor<0x9xf32>
    %11 = call @logaddexp_2(%9, %10) : (tensor<0x9xf32>, tensor<0x9xf32>) -> tensor<0x9xf32>
    %12 = "stablehlo.slice"(%5) {limit_indices = dense<[1, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<2x9xf32>) -> tensor<1x9xf32>
    %13 = stablehlo.concatenate %12, %11, dim = 0 : (tensor<1x9xf32>, tensor<0x9xf32>) -> tensor<1x9xf32>
    %14 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %15 = stablehlo.pad %13, %14, low = [0, 0], high = [1, 0], interior = [1, 0] : (tensor<1x9xf32>, tensor<f32>) -> tensor<2x9xf32>
    %16 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %17 = stablehlo.pad %8, %16, low = [1, 0], high = [0, 0], interior = [1, 0] : (tensor<1x9xf32>, tensor<f32>) -> tensor<2x9xf32>
    %18 = stablehlo.add %15, %17 : tensor<2x9xf32>
    %19 = "stablehlo.slice"(%18) {limit_indices = dense<[1, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<2x9xf32>) -> tensor<1x9xf32>
    %20 = "stablehlo.slice"(%2) {limit_indices = dense<[4, 9]> : tensor<2xi64>, start_indices = dense<[2, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<4x9xf32>) -> tensor<1x9xf32>
    %21 = call @logaddexp_3(%19, %20) : (tensor<1x9xf32>, tensor<1x9xf32>) -> tensor<1x9xf32>
    %22 = "stablehlo.slice"(%2) {limit_indices = dense<[1, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<4x9xf32>) -> tensor<1x9xf32>
    %23 = stablehlo.concatenate %22, %21, dim = 0 : (tensor<1x9xf32>, tensor<1x9xf32>) -> tensor<2x9xf32>
    %24 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %25 = stablehlo.pad %23, %24, low = [0, 0], high = [1, 0], interior = [1, 0] : (tensor<2x9xf32>, tensor<f32>) -> tensor<4x9xf32>
    %26 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %27 = stablehlo.pad %18, %26, low = [1, 0], high = [0, 0], interior = [1, 0] : (tensor<2x9xf32>, tensor<f32>) -> tensor<4x9xf32>
    %28 = stablehlo.add %25, %27 : tensor<4x9xf32>
    %29 = "stablehlo.slice"(%28) {limit_indices = dense<[3, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<4x9xf32>) -> tensor<3x9xf32>
    %30 = "stablehlo.slice"(%arg0) {limit_indices = dense<[8, 9]> : tensor<2xi64>, start_indices = dense<[2, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<8x9xf32>) -> tensor<3x9xf32>
    %31 = call @logaddexp_4(%29, %30) : (tensor<3x9xf32>, tensor<3x9xf32>) -> tensor<3x9xf32>
    %32 = "stablehlo.slice"(%arg0) {limit_indices = dense<[1, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<8x9xf32>) -> tensor<1x9xf32>
    %33 = stablehlo.concatenate %32, %31, dim = 0 : (tensor<1x9xf32>, tensor<3x9xf32>) -> tensor<4x9xf32>
    %34 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %35 = stablehlo.pad %33, %34, low = [0, 0], high = [1, 0], interior = [1, 0] : (tensor<4x9xf32>, tensor<f32>) -> tensor<8x9xf32>
    %36 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %37 = stablehlo.pad %28, %36, low = [1, 0], high = [0, 0], interior = [1, 0] : (tensor<4x9xf32>, tensor<f32>) -> tensor<8x9xf32>
    %38 = stablehlo.add %35, %37 : tensor<8x9xf32>
    return %38 : tensor<8x9xf32>
  }
  func.func private @logaddexp(%arg0: tensor<4x9xf32>, %arg1: tensor<4x9xf32>) -> tensor<4x9xf32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<4x9xf32>
    %1 = stablehlo.subtract %arg0, %arg1 : tensor<4x9xf32>
    %2 = stablehlo.compare  NE, %1, %1,  FLOAT : (tensor<4x9xf32>, tensor<4x9xf32>) -> tensor<4x9xi1>
    %3 = stablehlo.add %arg0, %arg1 : tensor<4x9xf32>
    %4 = stablehlo.abs %1 : tensor<4x9xf32>
    %5 = stablehlo.negate %4 : tensor<4x9xf32>
    %6 = stablehlo.exponential %5 : tensor<4x9xf32>
    %7 = stablehlo.log_plus_one %6 : tensor<4x9xf32>
    %8 = stablehlo.add %0, %7 : tensor<4x9xf32>
    %9 = stablehlo.select %2, %3, %8 : tensor<4x9xi1>, tensor<4x9xf32>
    return %9 : tensor<4x9xf32>
  }
  func.func private @logaddexp_0(%arg0: tensor<2x9xf32>, %arg1: tensor<2x9xf32>) -> tensor<2x9xf32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<2x9xf32>
    %1 = stablehlo.subtract %arg0, %arg1 : tensor<2x9xf32>
    %2 = stablehlo.compare  NE, %1, %1,  FLOAT : (tensor<2x9xf32>, tensor<2x9xf32>) -> tensor<2x9xi1>
    %3 = stablehlo.add %arg0, %arg1 : tensor<2x9xf32>
    %4 = stablehlo.abs %1 : tensor<2x9xf32>
    %5 = stablehlo.negate %4 : tensor<2x9xf32>
    %6 = stablehlo.exponential %5 : tensor<2x9xf32>
    %7 = stablehlo.log_plus_one %6 : tensor<2x9xf32>
    %8 = stablehlo.add %0, %7 : tensor<2x9xf32>
    %9 = stablehlo.select %2, %3, %8 : tensor<2x9xi1>, tensor<2x9xf32>
    return %9 : tensor<2x9xf32>
  }
  func.func private @logaddexp_1(%arg0: tensor<1x9xf32>, %arg1: tensor<1x9xf32>) -> tensor<1x9xf32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<1x9xf32>
    %1 = stablehlo.subtract %arg0, %arg1 : tensor<1x9xf32>
    %2 = stablehlo.compare  NE, %1, %1,  FLOAT : (tensor<1x9xf32>, tensor<1x9xf32>) -> tensor<1x9xi1>
    %3 = stablehlo.add %arg0, %arg1 : tensor<1x9xf32>
    %4 = stablehlo.abs %1 : tensor<1x9xf32>
    %5 = stablehlo.negate %4 : tensor<1x9xf32>
    %6 = stablehlo.exponential %5 : tensor<1x9xf32>
    %7 = stablehlo.log_plus_one %6 : tensor<1x9xf32>
    %8 = stablehlo.add %0, %7 : tensor<1x9xf32>
    %9 = stablehlo.select %2, %3, %8 : tensor<1x9xi1>, tensor<1x9xf32>
    return %9 : tensor<1x9xf32>
  }
  func.func private @logaddexp_2(%arg0: tensor<0x9xf32>, %arg1: tensor<0x9xf32>) -> tensor<0x9xf32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<0x9xf32>
    %1 = stablehlo.subtract %arg0, %arg1 : tensor<0x9xf32>
    %2 = stablehlo.compare  NE, %1, %1,  FLOAT : (tensor<0x9xf32>, tensor<0x9xf32>) -> tensor<0x9xi1>
    %3 = stablehlo.add %arg0, %arg1 : tensor<0x9xf32>
    %4 = stablehlo.abs %1 : tensor<0x9xf32>
    %5 = stablehlo.negate %4 : tensor<0x9xf32>
    %6 = stablehlo.exponential %5 : tensor<0x9xf32>
    %7 = stablehlo.log_plus_one %6 : tensor<0x9xf32>
    %8 = stablehlo.add %0, %7 : tensor<0x9xf32>
    %9 = stablehlo.select %2, %3, %8 : tensor<0x9xi1>, tensor<0x9xf32>
    return %9 : tensor<0x9xf32>
  }
  func.func private @logaddexp_3(%arg0: tensor<1x9xf32>, %arg1: tensor<1x9xf32>) -> tensor<1x9xf32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<1x9xf32>
    %1 = stablehlo.subtract %arg0, %arg1 : tensor<1x9xf32>
    %2 = stablehlo.compare  NE, %1, %1,  FLOAT : (tensor<1x9xf32>, tensor<1x9xf32>) -> tensor<1x9xi1>
    %3 = stablehlo.add %arg0, %arg1 : tensor<1x9xf32>
    %4 = stablehlo.abs %1 : tensor<1x9xf32>
    %5 = stablehlo.negate %4 : tensor<1x9xf32>
    %6 = stablehlo.exponential %5 : tensor<1x9xf32>
    %7 = stablehlo.log_plus_one %6 : tensor<1x9xf32>
    %8 = stablehlo.add %0, %7 : tensor<1x9xf32>
    %9 = stablehlo.select %2, %3, %8 : tensor<1x9xi1>, tensor<1x9xf32>
    return %9 : tensor<1x9xf32>
  }
  func.func private @logaddexp_4(%arg0: tensor<3x9xf32>, %arg1: tensor<3x9xf32>) -> tensor<3x9xf32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<3x9xf32>
    %1 = stablehlo.subtract %arg0, %arg1 : tensor<3x9xf32>
    %2 = stablehlo.compare  NE, %1, %1,  FLOAT : (tensor<3x9xf32>, tensor<3x9xf32>) -> tensor<3x9xi1>
    %3 = stablehlo.add %arg0, %arg1 : tensor<3x9xf32>
    %4 = stablehlo.abs %1 : tensor<3x9xf32>
    %5 = stablehlo.negate %4 : tensor<3x9xf32>
    %6 = stablehlo.exponential %5 : tensor<3x9xf32>
    %7 = stablehlo.log_plus_one %6 : tensor<3x9xf32>
    %8 = stablehlo.add %0, %7 : tensor<3x9xf32>
    %9 = stablehlo.select %2, %3, %8 : tensor<3x9xi1>, tensor<3x9xf32>
    return %9 : tensor<3x9xf32>
  }
}
