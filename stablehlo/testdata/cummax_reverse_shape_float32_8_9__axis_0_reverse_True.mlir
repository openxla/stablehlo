// RUN: stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt -inline | stablehlo-translate --interpret
// RUN: diff <(stablehlo-translate --deserialize %s.0_9_0.bc | stablehlo-opt) <(stablehlo-opt %s)
// RUN: diff <(stablehlo-translate --serialize --target=current %s | stablehlo-translate --deserialize | stablehlo-opt) <(stablehlo-opt %s)

module @jit_testcase {
  func.func public @main() -> tensor<i1> {
    %0 = call @inputs() : () -> tensor<8x9xf32>
    %1 = call @expected() : () -> tensor<8x9xf32>
    %2 = call @cummax(%0) : (tensor<8x9xf32>) -> tensor<8x9xf32>
    %3 = stablehlo.custom_call @check.eq(%2, %1) : (tensor<8x9xf32>, tensor<8x9xf32>) -> tensor<i1>
    return %3 : tensor<i1>
  }
  func.func private @inputs() -> tensor<8x9xf32> {
    %0 = stablehlo.constant dense<[[-1.82475162, -0.81886661, -2.64283347, -1.38385832, -2.85273075, 7.38511515, -0.261141062, -1.83053124, 7.04549121], [-1.33929646, -3.29696441, -1.31699264, -0.901468694, -0.591004193, 2.7991004, -0.599862218, 4.52632523, 1.29687142], [1.68646252, 2.18138647, 3.62680173, 3.00579572, -3.96342111, -8.619210e-01, -1.66585565, -1.32997668, 5.22923374], [0.469329208, 4.859950e+00, -2.70378709, -5.624260e+00, -0.675834537, -0.727197766, -6.08437538, 1.69178712, 1.95961654], [-4.28438044, 1.18180668, -0.542541265, -3.29918885, 0.250569969, 0.102437787, -2.2380178, 5.95544958, -1.15213025], [0.268647969, -2.83238792, -2.47498322, -0.360283613, -0.511627197, 0.394924492, -0.759667158, -3.41094351, -2.53735495], [2.86475372, -2.03218627, 6.16805649, -0.0468866453, 1.95897365, 1.36912525, 0.716629505, 0.116319381, -1.15647185], [2.12458253, 1.13289392, -1.62852824, 0.866116821, 0.148884416, 3.12622523, -4.00477123, -2.84748483, -1.18178427]]> : tensor<8x9xf32>
    return %0 : tensor<8x9xf32>
  }
  func.func private @expected() -> tensor<8x9xf32> {
    %0 = stablehlo.constant dense<[[2.86475372, 4.859950e+00, 6.16805649, 3.00579572, 1.95897365, 7.38511515, 0.716629505, 5.95544958, 7.04549121], [2.86475372, 4.859950e+00, 6.16805649, 3.00579572, 1.95897365, 3.12622523, 0.716629505, 5.95544958, 5.22923374], [2.86475372, 4.859950e+00, 6.16805649, 3.00579572, 1.95897365, 3.12622523, 0.716629505, 5.95544958, 5.22923374], [2.86475372, 4.859950e+00, 6.16805649, 0.866116821, 1.95897365, 3.12622523, 0.716629505, 5.95544958, 1.95961654], [2.86475372, 1.18180668, 6.16805649, 0.866116821, 1.95897365, 3.12622523, 0.716629505, 5.95544958, -1.15213025], [2.86475372, 1.13289392, 6.16805649, 0.866116821, 1.95897365, 3.12622523, 0.716629505, 0.116319381, -1.15647185], [2.86475372, 1.13289392, 6.16805649, 0.866116821, 1.95897365, 3.12622523, 0.716629505, 0.116319381, -1.15647185], [2.12458253, 1.13289392, -1.62852824, 0.866116821, 0.148884416, 3.12622523, -4.00477123, -2.84748483, -1.18178427]]> : tensor<8x9xf32>
    return %0 : tensor<8x9xf32>
  }
  func.func private @cummax(%arg0: tensor<8x9xf32>) -> tensor<8x9xf32> {
    %0 = stablehlo.reverse %arg0, dims = [0] : tensor<8x9xf32>
    %1 = "stablehlo.slice"(%0) {limit_indices = dense<[7, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<8x9xf32>) -> tensor<4x9xf32>
    %2 = "stablehlo.slice"(%0) {limit_indices = dense<[8, 9]> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<8x9xf32>) -> tensor<4x9xf32>
    %3 = stablehlo.maximum %1, %2 : tensor<4x9xf32>
    %4 = "stablehlo.slice"(%3) {limit_indices = dense<[3, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<4x9xf32>) -> tensor<2x9xf32>
    %5 = "stablehlo.slice"(%3) {limit_indices = dense<[4, 9]> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<4x9xf32>) -> tensor<2x9xf32>
    %6 = stablehlo.maximum %4, %5 : tensor<2x9xf32>
    %7 = "stablehlo.slice"(%6) {limit_indices = dense<[1, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<2x9xf32>) -> tensor<1x9xf32>
    %8 = "stablehlo.slice"(%6) {limit_indices = dense<[2, 9]> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<2x9xf32>) -> tensor<1x9xf32>
    %9 = stablehlo.maximum %7, %8 : tensor<1x9xf32>
    %10 = "stablehlo.slice"(%9) {limit_indices = dense<[0, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<1x9xf32>) -> tensor<0x9xf32>
    %11 = "stablehlo.slice"(%6) {limit_indices = dense<[2, 9]> : tensor<2xi64>, start_indices = dense<[2, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<2x9xf32>) -> tensor<0x9xf32>
    %12 = stablehlo.maximum %10, %11 : tensor<0x9xf32>
    %13 = "stablehlo.slice"(%6) {limit_indices = dense<[1, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<2x9xf32>) -> tensor<1x9xf32>
    %14 = stablehlo.concatenate %13, %12, dim = 0 : (tensor<1x9xf32>, tensor<0x9xf32>) -> tensor<1x9xf32>
    %15 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %16 = stablehlo.pad %14, %15, low = [0, 0], high = [1, 0], interior = [1, 0] : (tensor<1x9xf32>, tensor<f32>) -> tensor<2x9xf32>
    %17 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %18 = stablehlo.pad %9, %17, low = [1, 0], high = [0, 0], interior = [1, 0] : (tensor<1x9xf32>, tensor<f32>) -> tensor<2x9xf32>
    %19 = stablehlo.add %16, %18 : tensor<2x9xf32>
    %20 = "stablehlo.slice"(%19) {limit_indices = dense<[1, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<2x9xf32>) -> tensor<1x9xf32>
    %21 = "stablehlo.slice"(%3) {limit_indices = dense<[4, 9]> : tensor<2xi64>, start_indices = dense<[2, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<4x9xf32>) -> tensor<1x9xf32>
    %22 = stablehlo.maximum %20, %21 : tensor<1x9xf32>
    %23 = "stablehlo.slice"(%3) {limit_indices = dense<[1, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<4x9xf32>) -> tensor<1x9xf32>
    %24 = stablehlo.concatenate %23, %22, dim = 0 : (tensor<1x9xf32>, tensor<1x9xf32>) -> tensor<2x9xf32>
    %25 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %26 = stablehlo.pad %24, %25, low = [0, 0], high = [1, 0], interior = [1, 0] : (tensor<2x9xf32>, tensor<f32>) -> tensor<4x9xf32>
    %27 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %28 = stablehlo.pad %19, %27, low = [1, 0], high = [0, 0], interior = [1, 0] : (tensor<2x9xf32>, tensor<f32>) -> tensor<4x9xf32>
    %29 = stablehlo.add %26, %28 : tensor<4x9xf32>
    %30 = "stablehlo.slice"(%29) {limit_indices = dense<[3, 9]> : tensor<2xi64>, start_indices = dense<0> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<4x9xf32>) -> tensor<3x9xf32>
    %31 = "stablehlo.slice"(%0) {limit_indices = dense<[8, 9]> : tensor<2xi64>, start_indices = dense<[2, 0]> : tensor<2xi64>, strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<8x9xf32>) -> tensor<3x9xf32>
    %32 = stablehlo.maximum %30, %31 : tensor<3x9xf32>
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
