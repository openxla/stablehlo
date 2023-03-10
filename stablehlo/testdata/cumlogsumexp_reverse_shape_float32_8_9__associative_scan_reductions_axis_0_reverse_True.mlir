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
    %0 = stablehlo.constant dense<[[-0.401860863, 6.84116649, -2.70396471, 4.193640e+00, -0.19476369, -1.08573472, -1.74241042, 5.57146263, 2.91274118], [-0.86715275, -0.305921644, -0.638055384, -1.113860e+00, -4.34575033, 0.763507962, -3.6738193, 0.38010332, -2.69787216], [4.6862812, 0.440558195, 1.52119136, 2.82485414, -0.967525839, 2.80072331, -4.80217695, 2.29193902, -4.66183949], [2.12047744, 0.563039839, -1.93476629, -3.20223784, 1.54476547, -1.70058441, 3.6796248, 0.0298151355, -1.7952919], [-3.22694635, 3.05986691, 5.38926888, 5.48892689, -3.00548887, -5.49527025, 3.16498804, -0.553120315, -0.183059782], [1.01607084, 2.94962406, 1.756217, -7.54431247, 1.610290e+00, 0.0278498903, -0.24279888, -4.4796176, 7.46719217], [-3.819630e+00, -2.50333905, -6.224500e-01, -1.23910832, -0.763286411, -3.87448788, 5.15506124, -2.10532498, 2.40389943], [-2.700260e+00, 1.24534881, -1.38017201, -0.0741447583, -0.666994095, 2.56789637, -2.99275827, -0.495788246, -4.85855818]]> : tensor<8x9xf32>
    return %0 : tensor<8x9xf32>
  }
  func.func private @expected() -> tensor<8x9xf32> {
    %0 = stablehlo.constant dense<[[4.79384327, 6.89122486, 5.44215488, 5.78906393, 2.47956276, 3.50297284, 5.47088194, 5.62229204, 7.48450565], [4.78828764, 3.87173152, 5.44186497, 5.56238604, 2.40811682, 3.49275494, 5.47014475, 2.61770177, 7.47411203], [4.78478289, 3.85627818, 5.43957376, 5.5611248, 2.40694976, 3.42525911, 5.47003794, 2.50485277, 7.47407389], [2.41825366, 3.82287359, 5.41950035, 5.49411774, 2.37211394, 2.65844393, 5.4700036, 0.853415131, 7.47406864], [1.06164467, 3.7837224, 5.41886044, 5.49395037, 1.79727054, 2.64557052, 5.28740549, 0.275650024, 7.47397422], [1.04782546, 3.12036872, 1.88383019, 0.197791904, 1.78902972, 2.64527893, 5.15986538, -0.29809016, 7.4735012], [-2.41772699, 1.26862442, -0.238049895, 0.197357655, -0.0208345056, 2.56948781, 5.15535069, -0.313483179, 2.40460062], [-2.700260e+00, 1.24534881, -1.38017201, -0.0741447583, -0.666994095, 2.56789637, -2.99275827, -0.495788246, -4.85855818]]> : tensor<8x9xf32>
    return %0 : tensor<8x9xf32>
  }
  func.func private @cumlogsumexp(%arg0: tensor<8x9xf32>) -> tensor<8x9xf32> {
    %0 = stablehlo.reverse %arg0, dims = [0] : tensor<8x9xf32>
    %1 = "stablehlo.slice"(%0) {limit_indices = dense<[7, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<8x9xf32>) -> tensor<4x9xf32>
    %2 = "stablehlo.slice"(%0) {limit_indices = dense<[8, 9]> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<8x9xf32>) -> tensor<4x9xf32>
    %3 = call @logaddexp(%1, %2) : (tensor<4x9xf32>, tensor<4x9xf32>) -> tensor<4x9xf32>
    %4 = "stablehlo.slice"(%3) {limit_indices = dense<[3, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<4x9xf32>) -> tensor<2x9xf32>
    %5 = "stablehlo.slice"(%3) {limit_indices = dense<[4, 9]> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<4x9xf32>) -> tensor<2x9xf32>
    %6 = call @logaddexp_0(%4, %5) : (tensor<2x9xf32>, tensor<2x9xf32>) -> tensor<2x9xf32>
    %7 = "stablehlo.slice"(%6) {limit_indices = dense<[1, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<2x9xf32>) -> tensor<1x9xf32>
    %8 = "stablehlo.slice"(%6) {limit_indices = dense<[2, 9]> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<2x9xf32>) -> tensor<1x9xf32>
    %9 = call @logaddexp_1(%7, %8) : (tensor<1x9xf32>, tensor<1x9xf32>) -> tensor<1x9xf32>
    %10 = "stablehlo.slice"(%9) {limit_indices = dense<[0, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<1x9xf32>) -> tensor<0x9xf32>
    %11 = "stablehlo.slice"(%6) {limit_indices = dense<[2, 9]> : tensor<2xi64>, start_indices = dense<[2, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<2x9xf32>) -> tensor<0x9xf32>
    %12 = call @logaddexp_2(%10, %11) : (tensor<0x9xf32>, tensor<0x9xf32>) -> tensor<0x9xf32>
    %13 = "stablehlo.slice"(%6) {limit_indices = dense<[1, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<2x9xf32>) -> tensor<1x9xf32>
    %14 = stablehlo.concatenate %13, %12, dim = 0 : (tensor<1x9xf32>, tensor<0x9xf32>) -> tensor<1x9xf32>
    %15 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %16 = stablehlo.pad %14, %15, low = [0, 0], high = [1, 0], interior = [1, 0] : (tensor<1x9xf32>, tensor<f32>) -> tensor<2x9xf32>
    %17 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %18 = stablehlo.pad %9, %17, low = [1, 0], high = [0, 0], interior = [1, 0] : (tensor<1x9xf32>, tensor<f32>) -> tensor<2x9xf32>
    %19 = stablehlo.add %16, %18 : tensor<2x9xf32>
    %20 = "stablehlo.slice"(%19) {limit_indices = dense<[1, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<2x9xf32>) -> tensor<1x9xf32>
    %21 = "stablehlo.slice"(%3) {limit_indices = dense<[4, 9]> : tensor<2xi64>, start_indices = dense<[2, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<4x9xf32>) -> tensor<1x9xf32>
    %22 = call @logaddexp_3(%20, %21) : (tensor<1x9xf32>, tensor<1x9xf32>) -> tensor<1x9xf32>
    %23 = "stablehlo.slice"(%3) {limit_indices = dense<[1, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<4x9xf32>) -> tensor<1x9xf32>
    %24 = stablehlo.concatenate %23, %22, dim = 0 : (tensor<1x9xf32>, tensor<1x9xf32>) -> tensor<2x9xf32>
    %25 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %26 = stablehlo.pad %24, %25, low = [0, 0], high = [1, 0], interior = [1, 0] : (tensor<2x9xf32>, tensor<f32>) -> tensor<4x9xf32>
    %27 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %28 = stablehlo.pad %19, %27, low = [1, 0], high = [0, 0], interior = [1, 0] : (tensor<2x9xf32>, tensor<f32>) -> tensor<4x9xf32>
    %29 = stablehlo.add %26, %28 : tensor<4x9xf32>
    %30 = "stablehlo.slice"(%29) {limit_indices = dense<[3, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<4x9xf32>) -> tensor<3x9xf32>
    %31 = "stablehlo.slice"(%0) {limit_indices = dense<[8, 9]> : tensor<2xi64>, start_indices = dense<[2, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<8x9xf32>) -> tensor<3x9xf32>
    %32 = call @logaddexp_4(%30, %31) : (tensor<3x9xf32>, tensor<3x9xf32>) -> tensor<3x9xf32>
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
