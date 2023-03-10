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
    %0 = stablehlo.constant dense<[[-0.1621539, -7.20655059, -1.48743761, -3.26175737, 0.824775815, -3.37400675, 3.53261495, -0.395718634, -1.39859653], [-1.66363406, 1.94982815, -6.13506556, 2.094033, 1.404531, -1.412870e+00, 5.35104322, 2.26068807, -3.35760593], [2.85393643, -0.435547113, 8.76364136, -3.68753147, -1.15050614, 0.300769955, -1.14263666, 1.28852797, -3.67638898], [-3.44273448, 3.88409972, -1.08449399, -0.96742022, 6.07371044, -0.159820631, -0.439914495, 3.08994913, -0.986631631], [0.18370308, -1.03540123, -0.717823922, 0.0393680893, 4.20228577, -5.94003582, 3.03678393, -4.96846533, 3.95577836], [-2.14100313, 0.631120622, 1.69448602, -0.602440774, -3.55125904, 0.751082122, -2.65825057, -2.88465023, 1.90357387], [5.71682692, 3.60362124, -4.01821089, -2.40300751, 1.86689925, -1.83347273, 3.67490935, -0.996454775, 1.18008626], [-3.29687405, 2.27607799, 4.99171352, 2.07703233, 5.13094187, -5.940413, 2.63111234, -1.34229565, 1.70466495]]> : tensor<8x9xf32>
    return %0 : tensor<8x9xf32>
  }
  func.func private @expected() -> tensor<8x9xf32> {
    %0 = stablehlo.constant dense<[[-0.1621539, -1.612820e-01, 0.074182108, 0.109144866, 1.2228024, 1.23283577, 3.6281805, 3.64590573, 3.65232968], [-1.66363406, 1.97642958, 1.97672963, 2.7302475, 2.96580362, 2.97826767, 5.4401722, 5.480937, 5.48108196], [2.85393643, 2.89053178, 8.76645183, 8.76645565, 8.76650524, 8.76671504, 8.76676464, 8.767330e+00, 8.76733398], [-3.44273448, 3.88475704, 3.89168143, 3.89940882, 6.18138695, 6.18314791, 6.18447638, 6.22877693, 6.22951174], [0.18370308, 0.44259572, 0.715180397, 1.12645721, 4.24740362, 4.24744177, 4.5082674, 4.5083437, 4.96289825], [-2.14100313, 0.69177258, 2.00701857, 2.07801223, 2.08159709, 2.31614804, 2.32303691, 2.32849646, 2.83158445], [5.71682692, 5.83091402, 5.83096695, 5.83123207, 5.85003471, 5.85049534, 5.95804071, 5.95899439, 5.96736431], [-3.29687405, 2.279870e+00, 5.05601549, 5.105610e+00, 5.81150341, 5.81151104, 5.85223913, 5.852990e+00, 5.86865759]]> : tensor<8x9xf32>
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
