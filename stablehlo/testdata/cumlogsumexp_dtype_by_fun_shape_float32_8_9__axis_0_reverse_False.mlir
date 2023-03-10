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
    %0 = stablehlo.constant dense<[[3.42599034, -0.799378335, -2.37513781, 0.301761717, 1.97827518, -2.74963379, 3.70097136, 2.61135149, -1.04781473], [2.45742965, 0.802467585, 1.85314572, -0.806341052, -8.919590e+00, 3.24337029, 4.05146027, -0.496960133, -2.16944194], [6.35178947, -1.43565345, -2.13023829, -3.48172736, -5.38171959, -3.59886432, -3.14571524, -2.13203144, -6.33428335], [-4.96946239, 0.397940159, -1.6752094, 1.23312926, 9.501590e-01, -1.34421766, -8.53359508, 0.784738779, 0.822705686], [-4.1112628, -2.02338433, 0.636772811, 0.563314736, -0.863725841, 1.257339, 1.26683974, -0.175416201, -1.41990852], [-5.76511574, 1.79722083, 7.12741423, -0.374937624, 0.665109217, 5.01214695, 0.13961181, -1.78036129, 0.913279354], [7.04049063, 0.897011637, -2.21101904, -1.88830841, 0.7181589, 2.58707476, 0.999655425, -0.639652431, -0.00120709615], [1.38600123, -0.680391669, -2.2395885, 3.81347346, -1.23578382, -3.99782538, -4.591910e+00, -0.883973777, 0.90932697]]> : tensor<8x9xf32>
    return %0 : tensor<8x9xf32>
  }
  func.func private @expected() -> tensor<8x9xf32> {
    %0 = stablehlo.constant dense<[[3.42599034, -0.799378335, -2.37513781, 0.301761717, 1.97827518, -2.74963379, 3.70097136, 2.61135149, -1.04781473], [3.74780512, 0.986058473, 1.86761785, 0.587079585, 1.97829366, 3.24586344, 4.5846405, 2.65505862, -0.765836954], [6.42315912, 1.07110667, 1.88580632, 0.604032815, 1.97892964, 3.24692798, 4.58507967, 2.66336083, -0.7620278], [6.423170e+00, 1.48327637, 1.91382027, 1.66040361, 2.28453469, 3.25701809, 4.58508158, 2.80555487, 1.00918722], [6.42319679, 1.51283216, 2.15978885, 1.94846678, 2.3265655, 3.38398433, 4.62065744, 2.85505295, 1.09363544], [6.42320156, 2.35824943, 7.134350e+00, 2.04190207, 2.50040317, 5.19137192, 4.63191557, 2.86470819, 1.70066512], [7.47188663, 2.56684685, 7.13443708, 2.0613513, 2.6559186, 5.262720e+00, 4.65802813, 2.89433098, 1.86816216], [7.47415876, 2.60499144, 7.13452196, 3.97338367, 2.67612386, 5.26281548, 4.65812397, 2.91693521, 2.19266248]]> : tensor<8x9xf32>
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
