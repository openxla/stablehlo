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
    %0 = stablehlo.constant dense<[[1.41741407, 1.69045615, 3.49816585, 1.96060026, 0.342959732, 2.30391359, 1.54830503, -0.630049109, -2.89298415], [-0.142824933, 1.25122011, -0.630470395, -5.68282795, -1.45466101, 1.60190594, -0.973924994, 1.35886717, -4.34569836], [1.834131, -1.02774024, 2.87581921, -4.03832817, 0.0670142695, -3.43759251, 2.70158458, 1.74446332, 4.18799925], [-4.25361252, -1.71018529, -0.813623667, -2.22274256, -1.98764467, 0.0233882181, 0.456004828, 1.52560353, 0.555266798], [2.59989786, 1.5715152, 0.480428606, 5.71793604, 0.594837368, -1.62018442, -3.32945204, -2.20763612, 3.05773616], [5.08628511, -6.0236659, -2.20478606, -0.0936266929, 1.72348177, 3.02865577, -0.568275213, 1.02750921, -2.6691463], [-1.78086638, -2.87412548, 5.53764153, -3.45160174, -4.959320e+00, -2.33584309, 1.22951829, -3.45265961, -2.8652091], [-0.642917692, -0.77157098, 1.46999145, 6.28241729, 4.01627159, 0.887329697, 3.45534658, -0.635112583, -1.31859529]]> : tensor<8x9xf32>
    return %0 : tensor<8x9xf32>
  }
  func.func private @expected() -> tensor<8x9xf32> {
    %0 = stablehlo.constant dense<[[1.41741407, 2.25637245, 3.75192833, 3.90614057, 3.93409467, 4.11298895, 4.18711567, 4.19517279, 4.19600725], [-0.142824933, 1.47281837, 1.58797967, 1.5886749, 1.6352489, 2.31186342, 2.34859157, 2.66462731, 2.66552949], [1.834131, 1.88971865, 3.19283819, 3.19356155, 3.23649549, 3.23775792, 3.69833136, 3.83087134, 4.71844101], [-4.25361252, -1.63452458, -0.448956192, -0.292218059, -0.123724222, 6.456820e-01, 1.24848104, 2.0897584, 2.28496766], [2.59989786, 2.90560508, 2.99037123, 5.78126621, 5.78684234, 5.78744936, 5.78755903, 5.78789616, 5.85106707], [5.08628511, 5.086300e+00, 5.0869813, 5.092590e+00, 5.12643099, 5.2421937, 5.24518538, 5.25981045, 5.26017046], [-1.78086638, -1.49184334, 5.53852654, 5.53865147, 5.53867912, 5.53905916, 5.55240917, 5.5525322, 5.55275297], [-0.642917692, -0.0120295882, 1.67470872, 6.29234266, 6.39009237, 6.394160e+00, 6.44573497, 6.44657564, 6.447000e+00]]> : tensor<8x9xf32>
    return %0 : tensor<8x9xf32>
  }
  func.func private @cumlogsumexp(%arg0: tensor<8x9xf32>) -> tensor<8x9xf32> {
    %0 = "stablehlo.slice"(%arg0) {limit_indices = dense<8> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<[1, 2]> : tensor<2xi64>} : (tensor<8x9xf32>) -> tensor<8x4xf32>
    %1 = "stablehlo.slice"(%arg0) {limit_indices = dense<[8, 9]> : tensor<2xi64>, start_indices = dense<[0, 1]> : tensor<2xi64>, strides = dense<[1, 2]> : tensor<2xi64>} : (tensor<8x9xf32>) -> tensor<8x4xf32>
    %2 = call @logaddexp(%0, %1) : (tensor<8x4xf32>, tensor<8x4xf32>) -> tensor<8x4xf32>
    %3 = "stablehlo.slice"(%2) {limit_indices = dense<[8, 3]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<[1, 2]> : tensor<2xi64>} : (tensor<8x4xf32>) -> tensor<8x2xf32>
    %4 = "stablehlo.slice"(%2) {limit_indices = dense<[8, 4]> : tensor<2xi64>, start_indices = dense<[0, 1]> : tensor<2xi64>, strides = dense<[1, 2]> : tensor<2xi64>} : (tensor<8x4xf32>) -> tensor<8x2xf32>
    %5 = call @logaddexp_0(%3, %4) : (tensor<8x2xf32>, tensor<8x2xf32>) -> tensor<8x2xf32>
    %6 = "stablehlo.slice"(%5) {limit_indices = dense<[8, 1]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<[1, 2]> : tensor<2xi64>} : (tensor<8x2xf32>) -> tensor<8x1xf32>
    %7 = "stablehlo.slice"(%5) {limit_indices = dense<[8, 2]> : tensor<2xi64>, start_indices = dense<[0, 1]> : tensor<2xi64>, strides = dense<[1, 2]> : tensor<2xi64>} : (tensor<8x2xf32>) -> tensor<8x1xf32>
    %8 = call @logaddexp_1(%6, %7) : (tensor<8x1xf32>, tensor<8x1xf32>) -> tensor<8x1xf32>
    %9 = "stablehlo.slice"(%8) {limit_indices = dense<[8, 0]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<8x1xf32>) -> tensor<8x0xf32>
    %10 = "stablehlo.slice"(%5) {limit_indices = dense<[8, 2]> : tensor<2xi64>, start_indices = dense<[0, 2]> : tensor<2xi64>, strides = dense<[1, 2]> : tensor<2xi64>} : (tensor<8x2xf32>) -> tensor<8x0xf32>
    %11 = call @logaddexp_2(%9, %10) : (tensor<8x0xf32>, tensor<8x0xf32>) -> tensor<8x0xf32>
    %12 = "stablehlo.slice"(%5) {limit_indices = dense<[8, 1]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<8x2xf32>) -> tensor<8x1xf32>
    %13 = stablehlo.concatenate %12, %11, dim = 1 : (tensor<8x1xf32>, tensor<8x0xf32>) -> tensor<8x1xf32>
    %14 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %15 = stablehlo.pad %13, %14, low = [0, 0], high = [0, 1], interior = [0, 1] : (tensor<8x1xf32>, tensor<f32>) -> tensor<8x2xf32>
    %16 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %17 = stablehlo.pad %8, %16, low = [0, 1], high = [0, 0], interior = [0, 1] : (tensor<8x1xf32>, tensor<f32>) -> tensor<8x2xf32>
    %18 = stablehlo.add %15, %17 : tensor<8x2xf32>
    %19 = "stablehlo.slice"(%18) {limit_indices = dense<[8, 1]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<8x2xf32>) -> tensor<8x1xf32>
    %20 = "stablehlo.slice"(%2) {limit_indices = dense<[8, 4]> : tensor<2xi64>, start_indices = dense<[0, 2]> : tensor<2xi64>, strides = dense<[1, 2]> : tensor<2xi64>} : (tensor<8x4xf32>) -> tensor<8x1xf32>
    %21 = call @logaddexp_3(%19, %20) : (tensor<8x1xf32>, tensor<8x1xf32>) -> tensor<8x1xf32>
    %22 = "stablehlo.slice"(%2) {limit_indices = dense<[8, 1]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<8x4xf32>) -> tensor<8x1xf32>
    %23 = stablehlo.concatenate %22, %21, dim = 1 : (tensor<8x1xf32>, tensor<8x1xf32>) -> tensor<8x2xf32>
    %24 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %25 = stablehlo.pad %23, %24, low = [0, 0], high = [0, 1], interior = [0, 1] : (tensor<8x2xf32>, tensor<f32>) -> tensor<8x4xf32>
    %26 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %27 = stablehlo.pad %18, %26, low = [0, 1], high = [0, 0], interior = [0, 1] : (tensor<8x2xf32>, tensor<f32>) -> tensor<8x4xf32>
    %28 = stablehlo.add %25, %27 : tensor<8x4xf32>
    %29 = "stablehlo.slice"(%arg0) {limit_indices = dense<[8, 9]> : tensor<2xi64>, start_indices = dense<[0, 2]> : tensor<2xi64>, strides = dense<[1, 2]> : tensor<2xi64>} : (tensor<8x9xf32>) -> tensor<8x4xf32>
    %30 = call @logaddexp_4(%28, %29) : (tensor<8x4xf32>, tensor<8x4xf32>) -> tensor<8x4xf32>
    %31 = "stablehlo.slice"(%arg0) {limit_indices = dense<[8, 1]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<8x9xf32>) -> tensor<8x1xf32>
    %32 = stablehlo.concatenate %31, %30, dim = 1 : (tensor<8x1xf32>, tensor<8x4xf32>) -> tensor<8x5xf32>
    %33 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %34 = stablehlo.pad %32, %33, low = [0, 0], high = [0, 0], interior = [0, 1] : (tensor<8x5xf32>, tensor<f32>) -> tensor<8x9xf32>
    %35 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %36 = stablehlo.pad %28, %35, low = [0, 1], high = [0, 1], interior = [0, 1] : (tensor<8x4xf32>, tensor<f32>) -> tensor<8x9xf32>
    %37 = stablehlo.add %34, %36 : tensor<8x9xf32>
    return %37 : tensor<8x9xf32>
  }
  func.func private @logaddexp(%arg0: tensor<8x4xf32>, %arg1: tensor<8x4xf32>) -> tensor<8x4xf32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<8x4xf32>
    %1 = stablehlo.subtract %arg0, %arg1 : tensor<8x4xf32>
    %2 = stablehlo.compare  NE, %1, %1,  FLOAT : (tensor<8x4xf32>, tensor<8x4xf32>) -> tensor<8x4xi1>
    %3 = stablehlo.add %arg0, %arg1 : tensor<8x4xf32>
    %4 = stablehlo.abs %1 : tensor<8x4xf32>
    %5 = stablehlo.negate %4 : tensor<8x4xf32>
    %6 = stablehlo.exponential %5 : tensor<8x4xf32>
    %7 = stablehlo.log_plus_one %6 : tensor<8x4xf32>
    %8 = stablehlo.add %0, %7 : tensor<8x4xf32>
    %9 = stablehlo.select %2, %3, %8 : tensor<8x4xi1>, tensor<8x4xf32>
    return %9 : tensor<8x4xf32>
  }
  func.func private @logaddexp_0(%arg0: tensor<8x2xf32>, %arg1: tensor<8x2xf32>) -> tensor<8x2xf32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<8x2xf32>
    %1 = stablehlo.subtract %arg0, %arg1 : tensor<8x2xf32>
    %2 = stablehlo.compare  NE, %1, %1,  FLOAT : (tensor<8x2xf32>, tensor<8x2xf32>) -> tensor<8x2xi1>
    %3 = stablehlo.add %arg0, %arg1 : tensor<8x2xf32>
    %4 = stablehlo.abs %1 : tensor<8x2xf32>
    %5 = stablehlo.negate %4 : tensor<8x2xf32>
    %6 = stablehlo.exponential %5 : tensor<8x2xf32>
    %7 = stablehlo.log_plus_one %6 : tensor<8x2xf32>
    %8 = stablehlo.add %0, %7 : tensor<8x2xf32>
    %9 = stablehlo.select %2, %3, %8 : tensor<8x2xi1>, tensor<8x2xf32>
    return %9 : tensor<8x2xf32>
  }
  func.func private @logaddexp_1(%arg0: tensor<8x1xf32>, %arg1: tensor<8x1xf32>) -> tensor<8x1xf32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<8x1xf32>
    %1 = stablehlo.subtract %arg0, %arg1 : tensor<8x1xf32>
    %2 = stablehlo.compare  NE, %1, %1,  FLOAT : (tensor<8x1xf32>, tensor<8x1xf32>) -> tensor<8x1xi1>
    %3 = stablehlo.add %arg0, %arg1 : tensor<8x1xf32>
    %4 = stablehlo.abs %1 : tensor<8x1xf32>
    %5 = stablehlo.negate %4 : tensor<8x1xf32>
    %6 = stablehlo.exponential %5 : tensor<8x1xf32>
    %7 = stablehlo.log_plus_one %6 : tensor<8x1xf32>
    %8 = stablehlo.add %0, %7 : tensor<8x1xf32>
    %9 = stablehlo.select %2, %3, %8 : tensor<8x1xi1>, tensor<8x1xf32>
    return %9 : tensor<8x1xf32>
  }
  func.func private @logaddexp_2(%arg0: tensor<8x0xf32>, %arg1: tensor<8x0xf32>) -> tensor<8x0xf32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<8x0xf32>
    %1 = stablehlo.subtract %arg0, %arg1 : tensor<8x0xf32>
    %2 = stablehlo.compare  NE, %1, %1,  FLOAT : (tensor<8x0xf32>, tensor<8x0xf32>) -> tensor<8x0xi1>
    %3 = stablehlo.add %arg0, %arg1 : tensor<8x0xf32>
    %4 = stablehlo.abs %1 : tensor<8x0xf32>
    %5 = stablehlo.negate %4 : tensor<8x0xf32>
    %6 = stablehlo.exponential %5 : tensor<8x0xf32>
    %7 = stablehlo.log_plus_one %6 : tensor<8x0xf32>
    %8 = stablehlo.add %0, %7 : tensor<8x0xf32>
    %9 = stablehlo.select %2, %3, %8 : tensor<8x0xi1>, tensor<8x0xf32>
    return %9 : tensor<8x0xf32>
  }
  func.func private @logaddexp_3(%arg0: tensor<8x1xf32>, %arg1: tensor<8x1xf32>) -> tensor<8x1xf32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<8x1xf32>
    %1 = stablehlo.subtract %arg0, %arg1 : tensor<8x1xf32>
    %2 = stablehlo.compare  NE, %1, %1,  FLOAT : (tensor<8x1xf32>, tensor<8x1xf32>) -> tensor<8x1xi1>
    %3 = stablehlo.add %arg0, %arg1 : tensor<8x1xf32>
    %4 = stablehlo.abs %1 : tensor<8x1xf32>
    %5 = stablehlo.negate %4 : tensor<8x1xf32>
    %6 = stablehlo.exponential %5 : tensor<8x1xf32>
    %7 = stablehlo.log_plus_one %6 : tensor<8x1xf32>
    %8 = stablehlo.add %0, %7 : tensor<8x1xf32>
    %9 = stablehlo.select %2, %3, %8 : tensor<8x1xi1>, tensor<8x1xf32>
    return %9 : tensor<8x1xf32>
  }
  func.func private @logaddexp_4(%arg0: tensor<8x4xf32>, %arg1: tensor<8x4xf32>) -> tensor<8x4xf32> {
    %0 = stablehlo.maximum %arg0, %arg1 : tensor<8x4xf32>
    %1 = stablehlo.subtract %arg0, %arg1 : tensor<8x4xf32>
    %2 = stablehlo.compare  NE, %1, %1,  FLOAT : (tensor<8x4xf32>, tensor<8x4xf32>) -> tensor<8x4xi1>
    %3 = stablehlo.add %arg0, %arg1 : tensor<8x4xf32>
    %4 = stablehlo.abs %1 : tensor<8x4xf32>
    %5 = stablehlo.negate %4 : tensor<8x4xf32>
    %6 = stablehlo.exponential %5 : tensor<8x4xf32>
    %7 = stablehlo.log_plus_one %6 : tensor<8x4xf32>
    %8 = stablehlo.add %0, %7 : tensor<8x4xf32>
    %9 = stablehlo.select %2, %3, %8 : tensor<8x4xi1>, tensor<8x4xf32>
    return %9 : tensor<8x4xf32>
  }
}
