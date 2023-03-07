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
    %0 = stablehlo.constant dense<[[2.41221023, 0.99438095, -4.01280689, 0.250938356, 0.410634398, -2.00385904, -1.75666368, -3.21137071, 2.49663782], [-0.335578293, 0.143442348, -0.387393147, -3.3422215, 0.382177353, -1.40808523, 2.07854152, 3.1723032, 1.70976269], [2.28759193, 3.25584936, -1.50115633, -3.47315073, -2.62236238, -1.50488734, -3.38162303, 4.239630e+00, 4.6459446], [3.78663516, 3.00204134, 4.45821619, -0.115564391, 3.36583042, -4.95146084, -1.07542658, 1.86140442, 0.296632677], [-4.1012044, -1.42185235, -0.246377602, -3.09131742, 0.476042747, -1.03449953, 1.24812484, -1.45860612, 3.73737907], [1.96898139, 2.3736136, 3.57056355, 8.58149814, -1.82350624, 3.37253499, -2.19299293, -0.699369251, 0.682193637], [-4.434860e+00, -1.48574221, -2.50056148, 4.50787497, 5.52454853, -1.91370034, 0.180468634, -3.52148867, -0.00292339595], [-0.332320631, -3.31973743, 0.556404769, 6.76169968, 0.968424141, -7.61180925, 2.22561455, 4.88915968, 2.2878468]]> : tensor<8x9xf32>
    return %0 : tensor<8x9xf32>
  }
  func.func private @expected() -> tensor<8x9xf32> {
    %0 = stablehlo.constant dense<[[2.41221023, 0.99438095, -4.01280689, 0.250938356, 0.410634398, -2.00385904, -1.75666368, -3.21137071, 2.49663782], [2.47431087, 1.34996498, -0.361103684, 2.780780e-01, 1.08965433, -0.969097733, 2.09990859, 3.17399073, 2.87182593], [3.0784502, 3.39447236, -0.0836218297, 0.301295221, 1.11378896, -0.508382738, 2.1040628, 4.53565788, 4.80263472], [4.18711281, 3.91053224, 4.4688139, 0.807578563, 3.46584249, -0.496691644, 2.1448276, 4.60234261, 4.81361675], [4.1873641, 3.91535306, 4.47773218, 0.827640235, 3.51491594, -0.0367212892, 2.48693562, 4.60467196, 5.106940e+00], [4.29062891, 4.10928106, 4.81681919, 8.58192729, 3.51970792, 3.40506601, 2.49617243, 4.60963106, 5.11884594], [4.29079151, 4.11299038, 4.81748295, 8.59879302, 5.65090084, 3.40995288, 2.59029698, 4.60992527, 5.12479353], [4.30056572, 4.11358166, 4.83149147, 8.74659156, 5.66011429, 3.40996909, 3.11763573, 5.45240498, 5.18174505]]> : tensor<8x9xf32>
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
